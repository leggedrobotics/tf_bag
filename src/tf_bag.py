from __future__ import division
try:
    # Python 2
    from future_builtins import filter
except ImportError:
    # Python 3
    pass
import copy
import numpy as np
import rosbag
import rospy
import tf
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d


def transform_to_matrix(trans, quat):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(quat).as_dcm()
    matrix[:3, 3] = trans
    return matrix

class BagTfTransformer(object):
    """
    A transformer which transparently uses data recorded from rosbag on the /tf topic
    """

    def __init__(self, bag):
        """
        Create a new BagTfTransformer from an open rosbag or from a file path

        :param bag: an open rosbag or a file path to a rosbag file
        """
        if type(bag) == str:
            bag = rosbag.Bag(bag)
        self.tf_messages = sorted(
            (self._remove_slash_from_frames(tm) for m in bag if m.topic.strip("/") == 'tf' for tm in
             m.message.transforms),
            key=lambda tfm: tfm.header.stamp.to_nsec())
        self.tf_static_messages = sorted(
            (self._remove_slash_from_frames(tm) for m in bag if m.topic.strip("/") == 'tf_static' for tm in
             m.message.transforms),
            key=lambda tfm: tfm.header.stamp.to_nsec())

        self.tf_times = np.array(list((tfm.header.stamp.to_nsec() for tfm in self.tf_messages)))
        self.transformer = tf.TransformerROS()
        self.last_population_range = (rospy.Time(0), rospy.Time(0))
        self.all_frames = None
        self.all_transform_tuples = None
        self.static_transform_tuples = None

    @staticmethod
    def _remove_slash_from_frames(msg):
        msg.header.frame_id = msg.header.frame_id.strip("/")
        msg.child_frame_id = msg.child_frame_id.strip("/")
        return msg

    def getMessagesInTimeRange(self, min_time=None, max_time=None):
        """
        Returns all messages in the time range between two given ROS times

        :param min_time: the lower end of the desired time range (if None, the bag recording start time)
        :param max_time: the upper end of the desired time range (if None, the bag recording end time)
        :return: an iterator over the messages in the time range
        """
        import genpy
        if min_time is None:
            min_time = -float('inf')
        elif type(min_time) in (genpy.rostime.Time, rospy.rostime.Time):
            min_time = min_time.to_nsec()
        if max_time is None:
            max_time = float('inf')
        elif type(max_time) in (genpy.rostime.Time, rospy.rostime.Time):
            max_time = max_time.to_nsec()
        if max_time < min_time:
            raise ValueError('the minimum time should be lesser than the maximum time!')
        indices_in_range = np.where(np.logical_and(min_time < self.tf_times, self.tf_times < max_time))
        ret = (self.tf_messages[i] for i in indices_in_range[0])
        return ret

    def populateTransformerAtTime(self, target_time, buffer_length=10, lookahead=0.1):
        """
        Fills the buffer of the internal tf Transformer with the messages preceeding the given time

        :param target_time: the time at which the Transformer is going to be queried at next
        :param buffer_length: the length of the buffer, in seconds (default: 10, maximum for tf TransformerBuffer)
        """
        target_start_time = target_time - rospy.Duration(
            min(min(buffer_length, 10) - lookahead, target_time.to_sec()))  # max buffer length of tf Transformer
        target_end_time = target_time + rospy.Duration(lookahead)  # lookahead is there for numerical stability
        # otherwise, messages exactly around that time could be discarded
        previous_start_time, previous_end_time = self.last_population_range

        if target_start_time < previous_start_time:
            self.transformer.clear()  # or Transformer would ignore messages as old ones
            population_start_time = target_start_time
        else:
            population_start_time = max(target_start_time, previous_end_time)

        tf_messages_in_interval = self.getMessagesInTimeRange(population_start_time, target_end_time)
        for m in tf_messages_in_interval:
            self.transformer.setTransform(m)
        for st_tfm in self.tf_static_messages:
            st_tfm.header.stamp = target_time
            self.transformer._buffer.set_transform_static(st_tfm, "default_authority")

        self.last_population_range = (target_start_time, target_end_time)

    def getTimeAtPercent(self, percent):
        """
        Returns the ROS time at the given point in the time range

        :param percent: the point in the recorded time range for which the ROS time is desired
        :return:
        """
        start_time, end_time = self.getStartTime(), self.getEndTime()
        time_range = (end_time - start_time).to_sec()
        ret = start_time + rospy.Duration(time_range * float(percent / 100))
        return ret

    def _filterMessages(self, orig_frame=None, dest_frame=None, start_time=None, end_time=None, reverse=False):
        if reverse:
            messages = reversed(self.tf_messages)
        else:
            messages = self.tf_messages

        if orig_frame:
            messages = filter(lambda m: m.header.frame_id == orig_frame, messages)
        if dest_frame:
            messages = filter(lambda m: m.child_frame_id == dest_frame, messages)
        if start_time:
            messages = filter(lambda m: m.header.stamp > start_time, messages)
        if end_time:
            messages = filter(lambda m: m.header.stamp < end_time, messages)
        return messages

    def getTransformMessagesWithFrame(self, frame, start_time=None, end_time=None, reverse=False):
        """
        Returns all transform messages with given frame as source or target frame

        :param frame: the tf frame of interest
        :param start_time: the time at which the messages should start; if None, all recorded messages
        :param end_time: the time at which the messages should end; if None, all recorded messages
        :param reverse: if True, the messages will be provided in reversed order
        :return: an iterator over the messages respecting the criteria
        """
        for m in self._filterMessages(start_time=start_time, end_time=end_time, reverse=reverse):
            if m.header.frame_id == frame or m.child_frame_id == frame:
                yield m

    def getFrameStrings(self):
        """
        Returns the IDs of all tf frames

        :return: a set containing all known tf frame IDs
        """
        if self.all_frames is None:
            ret = set()
            for m in self.tf_messages:
                ret.add(m.header.frame_id)
                ret.add(m.child_frame_id)
            self.all_frames = ret

        return self.all_frames

    def getTransformFrameTuples(self):
        """
        Returns all pairs of directly connected tf frames

        :return: a set containing all known tf frame pairs
        """
        if self.all_transform_tuples is None:
            ret = set()
            for m in self.tf_messages:
                ret.add((m.header.frame_id, m.child_frame_id))
            for m in self.tf_static_messages:
                ret.add((m.header.frame_id, m.child_frame_id))
            self.all_transform_tuples = ret
            self.static_transform_tuples = {(m.header.frame_id, m.child_frame_id) for m in self.tf_static_messages}

        return self.all_transform_tuples

    def getTransformGraphInfo(self, time=None):
        """
        Returns the output of TfTransformer.allFramesAsDot() at a given point in time

        :param time: the ROS time at which tf should be queried; if None, it will be the buffer middle time
        :return: A string containing information about the tf tree
        """
        if time is None:
            time = self.getTimeAtPercent(50)
        self.populateTransformerAtTime(time)
        return self.transformer.allFramesAsDot()

    def getStartTime(self):
        """
        Returns the time of the first tf message in the buffer

        :return: the ROS time of the first tf message in the buffer
        """
        return self.tf_messages[0].header.stamp

    def getEndTime(self):
        """
        Returns the time of the last tf message in the buffer

        :return: the ROS time of the last tf message in the buffer
        """
        return self.tf_messages[-1].header.stamp

    @staticmethod
    def _getTimeFromTransforms(transforms):
        return (t.header.stamp for t in transforms)

    def getAverageUpdateFrequency(self, orig_frame, dest_frame, start_time=None, end_time=None):
        """
        Computes the average time between two tf messages directly connecting two given frames

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :param end_time: the last time at which the messages should be considered; if None, all recorded messages
        :return: the average transform update frequency
        """
        messages = self._filterMessages(orig_frame=orig_frame, dest_frame=dest_frame,
                                        start_time=start_time, end_time=end_time)
        message_times = BagTfTransformer._getTimeFromTransforms(messages)
        message_times = np.array(message_times)
        average_delta = (message_times[1:] - message_times[:-1]).mean()
        return average_delta

    def getTransformUpdateTimes(self, orig_frame, dest_frame, trigger_orig_frame=None, trigger_dest_frame=None,
                                start_time=None, end_time=None, reverse=False):
        """
        Returns the times at which the transform between two frames was updated.

        If the two frames are not directly connected, two directly connected "trigger frames" must be provided.
        The result will be then the update times of the transform between the two frames, but will start at the
        time when the entire transformation chain is complete.

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :param end_time: the last time at which the messages should be considered; if None, all recorded messages
        :param reverse: if True, the times will be provided in reversed order
        :return: an iterator over the times at which the transform is updated
        """
        trigger_frames_were_provided = trigger_orig_frame is not None or trigger_dest_frame is not None
        if trigger_orig_frame is None:
            trigger_orig_frame = orig_frame
        if trigger_dest_frame is None:
            trigger_dest_frame = dest_frame
        if (trigger_dest_frame, trigger_orig_frame) in self.getTransformFrameTuples():
            trigger_orig_frame, trigger_dest_frame = trigger_dest_frame, trigger_orig_frame
        updates = list(self._filterMessages(orig_frame=trigger_orig_frame, dest_frame=trigger_dest_frame,
                                            start_time=start_time, end_time=end_time, reverse=reverse))
        if not updates:
            if trigger_frames_were_provided:
                raise RuntimeError('the provided trigger frames ({}->{}) must be directly connected!'
                                   .format(trigger_orig_frame, trigger_dest_frame))
            else:
                raise RuntimeError('the two frames ({}->{}) are not directly connected! you must provide \
                 directly connected "trigger frames"'.format(trigger_orig_frame, trigger_dest_frame))
        first_update_time = self.waitForTransform(orig_frame, dest_frame, start_time=start_time)
        return (t for t in BagTfTransformer._getTimeFromTransforms(updates) if t > first_update_time)

    def waitForTransform(self, orig_frame, dest_frame, start_time=None, duration=None):
        """
        Returns the first time for which at least a tf message is available for the whole chain between \
        the two provided frames

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :return: the ROS time at which the transform is available
        """
        if orig_frame == dest_frame:
            return self.tf_messages[0].header.stamp
        if start_time is not None:
            messages = filter(lambda m: m.header.stamp > start_time, self.tf_messages)
        else:
            messages = self.tf_messages
        missing_transforms = set(self.getChainTuples(orig_frame, dest_frame)) - self.static_transform_tuples
        message = messages.__iter__()
        ret = rospy.Time(0)
        try:
            while missing_transforms:
                m = next(message)
                if (m.header.frame_id, m.child_frame_id) in missing_transforms:
                    missing_transforms.remove((m.header.frame_id, m.child_frame_id))
                    ret = max(ret, m.header.stamp)
                if (m.child_frame_id, m.header.frame_id) in missing_transforms:
                    missing_transforms.remove((m.child_frame_id, m.header.frame_id))
                    ret = max(ret, m.header.stamp)
        except StopIteration:
            raise ValueError('Transform not found between {} and {}'.format(orig_frame, dest_frame))
        return ret

    def lookupTransform(self, orig_frame, dest_frame, time, latest=False, method="simple", odom_world=None, odom_body=None, return_se3=False):
        """
        Returns the transform between the two provided frames at the given time

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param time: the first time at which the messages should be considered; if None, all recorded messages
        :return: the ROS time at which the transform is available
        """
        if orig_frame == dest_frame:
            return (0, 0, 0), (0, 0, 0, 1)

        if latest:
            try:
                time = self.tf_static_messages[0].header.stamp
            except:
                time = self.getStartTime()

        self.populateTransformerAtTime(time)


        if method == "simple":
            try:
                common_time = self.transformer.getLatestCommonTime(orig_frame, dest_frame)
            except:
                print('Could not find the transformation {} -> {} in the 10 seconds before time {}'
                                .format(orig_frame, dest_frame, time))
                if return_se3:
                    return None
                return (None,None)

            res = self.transformer.lookupTransform(orig_frame, dest_frame, common_time)

        elif method == "interpolate_linear":
            # Here we use the orig_frame to look all transforms.
            # Current implementation is brittle and may fail in different settings when more odom sources are published.

            dt = 0.2
            pre = time - rospy.Duration(dt)
            post = time + rospy.Duration(dt)
            res = [
                f for f in self.getTransformMessagesWithFrame(orig_frame, start_time=pre, end_time=post, reverse=False)
            ]
            after_msg = None
            for msg in res:
                if msg.header.stamp <= time:
                    before_msg = msg
                elif msg.header.stamp > time and after_msg is None:
                    after_msg = msg
                    break
            res = [before_msg, after_msg]

            # Get the two transforms
            tf1 = self.transformer.lookupTransform(orig_frame, dest_frame, res[0].header.stamp)
            tf2 = self.transformer.lookupTransform(orig_frame, dest_frame, res[1].header.stamp)

            # Extract timestamps - deleting the d for numerical stabiltiy but not needed
            d = rospy.Duration(res[0].header.stamp.secs)
            t1 = (res[0].header.stamp - d).to_sec()
            t2 = (res[1].header.stamp - d).to_sec()
            target_time = (time - d).to_sec()

            # Extract translations and rotations from transforms
            trans1 = np.array(tf1[0])
            trans2 = np.array(tf2[0])

            quat1 = np.array(tf1[1])
            quat2 = np.array(tf2[1])

            # Create arrays for interpolation
            timestamps = np.array([t1, t2])
            rotations = R.from_quat([quat1, quat2])
            translations = np.array([trans1, trans2])

            # SLERP interpolation for rotations
            slerp_interpolator = Slerp(timestamps, rotations)
            interpolated_rotation = slerp_interpolator([target_time])

            # Linear interpolation for translations
            translation_interpolator = interp1d(
                timestamps,
                translations,
                kind="linear",
                axis=0,
                bounds_error=False,
                fill_value=(trans1, trans2),  # Clamp to boundary values
            )
            interpolated_translation = translation_interpolator([target_time])[0]
            interpolated_quat = interpolated_rotation.as_quat()
            res = (interpolated_translation, interpolated_quat)

        elif method == "interpolate_odom_frame":
            # Very inefficent way to fetch the previous world frame transform
            dt = 0.2

            pre = time - rospy.Duration(dt)
            post = time + rospy.Duration(dt)
            res = [
                f for f in self.getTransformMessagesWithFrame(orig_frame, start_time=pre, end_time=post, reverse=False)
            ]
            after_msg = None
            for msg in res:
                if msg.header.stamp <= time:
                    before_msg = msg
                elif msg.header.stamp > time and after_msg is None:
                    after_msg = msg
                    break

            res = [before_msg, after_msg]

            # Get the transform available from previous orig_frame - assumes then the odom topic is available at high frequency
            tf1 = self.transformer.lookupTransform(orig_frame, dest_frame, res[0].header.stamp)
            interpol_odom_start = self.transformer.lookupTransform(odom_body, odom_world, res[0].header.stamp)
            interpol_odom_dest = self.transformer.lookupTransform(odom_body, odom_world, time)

            odom_start_matrix = transform_to_matrix(interpol_odom_start[0], interpol_odom_start[1])
            odom_dest_matrix = transform_to_matrix(interpol_odom_dest[0], interpol_odom_dest[1])

            # Compute the relative transform from start to dest
            odom_relative_matrix = np.linalg.inv(odom_start_matrix) @ odom_dest_matrix

            # Convert tf1 to matrix
            tf1_matrix = transform_to_matrix(tf1[0], tf1[1])

            # Apply the relative odom transform to tf1
            tf1_corrected_matrix = tf1_matrix @ odom_relative_matrix

            # Convert back to (trans, quat) format
            tf1_corrected_trans = tf1_corrected_matrix[:3, 3]
            tf1_corrected_quat = R.from_dcm(tf1_corrected_matrix[:3, :3]).as_quat()

            # Update tf1 with corrected values
            res = (tf1_corrected_trans, tf1_corrected_quat)


        if return_se3:
            return transform_to_matrix(res[0], res[1])
        return res

    def lookupTransformWhenTransformUpdates(self, orig_frame, dest_frame,
                                            trigger_orig_frame=None, trigger_dest_frame=None,
                                            start_time=None, end_time=None):
        """
        Returns the transform between two frames every time it updates

        If the two frames are not directly connected, two directly connected "trigger frames" must be provided.
        The result will be then sampled at the update times of the transform between the two frames.

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :param start_time: the first time at which the messages should be considered; if None, all recorded messages
        :param end_time: the last time at which the messages should be considered; if None, all recorded messages
        :return: an iterator over tuples containing the update time and the transform
        """
        update_times = self.getTransformUpdateTimes(orig_frame, dest_frame,
                                                    trigger_orig_frame=trigger_orig_frame,
                                                    trigger_dest_frame=trigger_dest_frame,
                                                    start_time=start_time, end_time=end_time)
        ret = ((t, self.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame, time=t)) for t in update_times)
        return ret

    def getFrameAncestors(self, frame, early_stop_frame=None):
        """
        Returns the ancestor frames of the given tf frame, until the tree root

        :param frame: ID of the tf frame of interest
        :param early_stop_frame: if not None, stop when this frame is encountered
        :return: a list representing the succession of frames from the tf tree root to the provided one
        """
        frame_chain = [frame]
        chain_link = list(filter(lambda tt: tt[1] == frame, self.getTransformFrameTuples()))
        while chain_link and frame_chain[-1] != early_stop_frame:
            frame_chain.append(chain_link[0][0])
            chain_link = list(filter(lambda tt: tt[1] == frame_chain[-1], self.getTransformFrameTuples()))
        return list(reversed(frame_chain))

    def getChain(self, orig_frame, dest_frame):
        """
        Returns the chain of frames between two frames

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :return: a list representing the succession of frames between the two passed as argument
        """
        # transformer.chain is apparently bugged
        orig_ancestors = self.getFrameAncestors(orig_frame, early_stop_frame=dest_frame)
        if orig_ancestors[0] == dest_frame:
            return orig_ancestors
        dest_ancestors = self.getFrameAncestors(dest_frame, early_stop_frame=orig_frame)
        if dest_ancestors[0] == orig_frame:
            return dest_ancestors
        if orig_ancestors[0] == dest_ancestors[-1]:
            return list(reversed(dest_ancestors)) + orig_ancestors[1:]
        if dest_ancestors[0] == orig_ancestors[-1]:
            return list(reversed(orig_ancestors)) + dest_ancestors[1:]
        while len(dest_ancestors) > 0 and orig_ancestors[0] == dest_ancestors[0]:
            if len(orig_ancestors) > 1 and len(dest_ancestors) > 1 and orig_ancestors[1] == dest_ancestors[1]:
                orig_ancestors.pop(0)
            dest_ancestors.pop(0)
        return list(reversed(orig_ancestors)) + dest_ancestors

    def getChainTuples(self, orig_frame, dest_frame):
        """
        Returns the chain of frame pairs representing the transforms connecting two frames

        :param orig_frame: the source tf frame of the transform chain of interest
        :param dest_frame: the target tf frame of the transform chain of interest
        :return: a list of frame ID pairs representing the succession of transforms between the frames passed as argument
        """
        chain = self.getChain(orig_frame, dest_frame)
        return zip(chain[:-1], chain[1:])

    @staticmethod
    def averageTransforms(transforms):
        """
        Computes the average transform over the ones passed as argument

        :param transforms: a list of transforms
        :return: a transform having the average value
        """
        if not transforms:
            raise RuntimeError('requested average of an empty vector of transforms')
        transforms = list(transforms)
        translations = np.array([t[0] for t in transforms])
        quaternions = np.array([t[1] for t in transforms])
        mean_translation = translations.mean(axis=0).tolist()
        mean_quaternion = quaternions.mean(axis=0)  # I know, it is horrible.. but for small rotations shouldn't matter
        mean_quaternion = (mean_quaternion / np.linalg.norm(mean_quaternion)).tolist()
        return mean_translation, mean_quaternion

    def averageTransformOverTime(self, orig_frame, dest_frame, start_time, end_time,
                                 trigger_orig_frame=None, trigger_dest_frame=None):
        """
        Computes the average value of the transform between two frames

        If the two frames are not directly connected, two directly connected "trigger frames" must be provided.
        The result will be then sampled at the update times of the transform between the two frames, but will start at the
        time when the entire transformation chain is complete.

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the start time of the averaging time range
        :param end_time: the end time of the averaging time range
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :return: the average value of the transformation over the specified time range
        """
        if orig_frame == dest_frame:
            return (0, 0, 0), (0, 0, 0, 1)
        update_times = self.getTransformUpdateTimes(orig_frame=orig_frame, dest_frame=dest_frame,
                                                    start_time=start_time, end_time=end_time,
                                                    trigger_orig_frame=trigger_orig_frame,
                                                    trigger_dest_frame=trigger_dest_frame)
        target_transforms = (self.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame, time=t)
                             for t in update_times)
        return self.averageTransforms(target_transforms)

    def replicateTransformOverTime(self, transf, orig_frame, dest_frame, frequency, start_time=None, end_time=None):
        """
        Adds a new transform to the graph with the specified value

        This can be useful to add calibration a-posteriori.

        :param transf: value of the transform
        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param frequency: frequency at which the transform should be published
        :param start_time: the time the transform should be published from
        :param end_time: the time the transform should be published until
        :return:
        """
        if start_time is None:
            start_time = self.getStartTime()
        if end_time is None:
            end_time = self.getEndTime()
        transl, quat = transf
        time_delta = rospy.Duration(1 / frequency)

        t_msg = TransformStamped(header=Header(frame_id=orig_frame),
                                 child_frame_id=dest_frame,
                                 transform=Transform(translation=Vector3(*transl), rotation=Quaternion(*quat)))

        def createMsg(time_nsec):
            time = rospy.Time(time_nsec / 1000000000)
            t_msg2 = copy.deepcopy(t_msg)
            t_msg2.header.stamp = time
            return t_msg2

        new_msgs = [createMsg(t) for t in range(start_time.to_nsec(), end_time.to_nsec(), time_delta.to_nsec())]
        self.tf_messages += new_msgs
        self.tf_messages.sort(key=lambda tfm: tfm.header.stamp.to_nsec())
        self.tf_times = np.array(list((tfm.header.stamp.to_nsec() for tfm in self.tf_messages)))
        self.all_transform_tuples.add((orig_frame, dest_frame))

    def processTransform(self, callback, orig_frame, dest_frame,
                         trigger_orig_frame=None, trigger_dest_frame=None, start_time=None, end_time=None):
        """
        Looks up the transform between two frames and forwards it to a callback at each update

        :param callback: a function taking two arguments (the time and the transform as a tuple of translation and rotation)
        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param start_time: the start time of the time range
        :param end_time: the end time of the time range
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :return: an iterator over the result of calling the callback with the looked up transform as argument
        """
        times = self.getTransformUpdateTimes(orig_frame, dest_frame, trigger_orig_frame, trigger_dest_frame,
                                             start_time=start_time, end_time=end_time)
        transforms = [(t, self.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame, time=t)) for t in times]
        for time, transform in transforms:
            yield callback(time, transform)

    def plotTranslation(self, orig_frame, dest_frame, axis=None,
                        trigger_orig_frame=None, trigger_dest_frame=None, start_time=None, end_time=None,
                        fig=None, ax=None, color='blue'):
        """
        Creates a 2D or 3D plot of the trajectory described by the values of the translation of the transform over time

        :param orig_frame: the source tf frame of the transform of interest
        :param dest_frame: the target tf frame of the transform of interest
        :param axis: if None, the plot will be 3D; otherwise, it should be 'x', 'y', or 'z': the value will be plotted over time
        :param trigger_orig_frame: the source tf frame of a transform in the chain, directly connected to trigger_dest_frame
        :param trigger_dest_frame: the target tf frame of a transform in the chain, directly connected to trigger_orig_frame
        :param start_time: the start time of the time range
        :param end_time: the end time of the time range
        :param fig: if provided, the Matplotlib figure will be reused; otherwise a new one will be created
        :param ax: if provided, the Matplotlib axis will be reused; otherwise a new one will be created
        :param color: the color of the line
        :return:
        """
        import matplotlib.pyplot as plt
        if axis is None:
            # 3D
            from mpl_toolkits.mplot3d import Axes3D
            translation_data = np.array(list(self.processTransform(lambda t, tr: (tr[0]),
                                                                   orig_frame=orig_frame, dest_frame=dest_frame,
                                                                   trigger_orig_frame=trigger_orig_frame,
                                                                   trigger_dest_frame=trigger_dest_frame,
                                                                   start_time=start_time, end_time=end_time)))
            if fig is None:
                fig = plt.figure()
            if ax is None:
                ax = fig.add_subplot(111, projection='3d')

            ax.scatter(
                translation_data[:, 0],
                translation_data[:, 1],
                translation_data[:, 2],
                c=color
            )
            return ax, fig
        else:
            translation_data = np.array(list(self.processTransform(lambda t, tr: (t.to_nsec(), tr[0][axis]),
                                                                   orig_frame=orig_frame, dest_frame=dest_frame,
                                                                   trigger_orig_frame=trigger_orig_frame,
                                                                   trigger_dest_frame=trigger_dest_frame,
                                                                   start_time=start_time, end_time=end_time)))
            if fig is None:
                fig = plt.figure()
            if ax is None:
                ax = fig.add_subplot(111)
            ax.plot(translation_data[:, 0], translation_data[:, 1], color=color)
            return ax, fig
