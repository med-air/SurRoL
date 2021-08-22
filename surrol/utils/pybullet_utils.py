"""
This file is modified from
https://github.com/columbia-robovision/decentralized-multiarm/blob/main/environment/rrt/pybullet_utils.py
"""
import time
import numpy as np
import pybullet as p
import pybullet_data
from collections import namedtuple, defaultdict, deque
from itertools import product, combinations

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI
MAX_DISTANCE = 0


def step(duration=1.0):
    for i in range(int(duration * 240)):
        p.stepSimulation()


def step_real(duration=1.0):
    for i in range(int(duration * 240)):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def split_7d(pose):
    return [list(pose[:3]), list(pose[3:])]


def merge_pose_2d(pose):
    return pose[0] + pose[1]


def get_euler_from_quaternion(quaternion):
    return list(p.getEulerFromQuaternion(quaternion))


def get_quaternion_from_euler(euler):
    return list(p.getQuaternionFromEuler(euler))


def configure_pybullet(rendering=False, debug=False, yaw=50.0, pitch=-35.0, dist=1.2, target=(0.0, 0.0, 0.0)):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetSimulation()
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.8)


# Constraints

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild',
                                               'maxAppliedForce'])


def remove_all_constraints():
    for cid in get_constraint_ids():
        p.removeConstraint(cid)


def get_constraint_ids():
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return sorted([p.getConstraintUniqueId(i) for i in range(p.getNumConstraints())])


def get_constraint_info(constraint):
    # there are four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint)[:11])


# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute',  # 0
    p.JOINT_PRISMATIC: 'prismatic',  # 1
    p.JOINT_SPHERICAL: 'spherical',  # 2
    p.JOINT_PLANAR: 'planar',  # 3
    p.JOINT_FIXED: 'fixed',  # 4
    p.JOINT_POINT2POINT: 'point2point',  # 5
    p.JOINT_GEAR: 'gear',  # 6
}


def get_num_joints(body):
    return p.getNumJoints(body)


def get_joints(body):
    return list(range(get_num_joints(body)))


def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name


JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])


def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint))


def get_joints_info(body, joints):
    return [JointInfo(*p.getJointInfo(body, joint)) for joint in joints]


def get_joint_name(body, joint):
    return get_joint_info(body, joint).jointName.decode('UTF-8')


def get_joint_names(body):
    return [get_joint_name(body, joint) for joint in get_joints(body)]


def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)


def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True


def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)


JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint))


def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition


def get_joint_torque(body, joint):
    return get_joint_state(body, joint).appliedJointMotorTorque


def get_joint_positions(body, joints=None):
    return list(get_joint_position(body, joint) for joint in joints)


def set_joint_position(body, joint, value):
    p.resetJointState(body, joint, value)


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)


def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))


def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)


def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))


def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType


def is_movable(body, joint):
    return get_joint_type(body, joint) != p.JOINT_FIXED


def get_movable_joints(body):  # 45 / 87 on pr2
    return [joint for joint in get_joints(body) if is_movable(body, joint)]


def joint_from_movable(body, index):
    return get_joints(body)[index]


def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit


def get_joint_limits(body, joint):
    """
    Obtain the limits of a single joint
    :param body: int
    :param joint: int
    :return: (int, int), lower limit and upper limit
    """
    if is_circular(body, joint):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_joints_limits(body, joints):
    """
    Obtain the limits of a set of joints
    :param body: int
    :param joints: array type
    :return: a tuple of 2 arrays - lower limit and higher limit
    """
    lower_limit = []
    upper_limit = []
    for joint in joints:
        lower_limit.append(get_joint_info(body, joint).jointLowerLimit)
        upper_limit.append(get_joint_info(body, joint).jointUpperLimit)
    return lower_limit, upper_limit


def get_min_limit(body, joint):
    return get_joint_limits(body, joint)[0]


def get_max_limit(body, joint):
    return get_joint_limits(body, joint)[1]


def get_max_velocity(body, joint):
    return get_joint_info(body, joint).jointMaxVelocity


def get_max_force(body, joint):
    return get_joint_info(body, joint).jointMaxForce


def get_joint_q_index(body, joint):
    return get_joint_info(body, joint).qIndex


def get_joint_v_index(body, joint):
    return get_joint_info(body, joint).uIndex


def get_joint_axis(body, joint):
    return get_joint_info(body, joint).jointAxis


def get_joint_parent_frame(body, joint):
    joint_info = get_joint_info(body, joint)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def violates_limit(body, joint, value):
    if not is_circular(body, joint):
        lower, upper = get_joint_limits(body, joint)
        if (value < lower) or (upper < value):
            return True
    return False


def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def wrap_joint(body, joint, value):
    if is_circular(body, joint):
        return wrap_angle(value)
    return value


def get_difference_fn(body, joints):
    def fn(q2, q1):
        difference = []
        for joint, value2, value1 in zip(joints, q2, q1):
            difference.append(circular_difference(value2, value1)
                              if is_circular(body, joint)
                              else (np.array(value2) - np.array(value1)))
        return list(difference)

    return fn


def get_distance_fn(body, joints, weights=None):
    if weights is None:
        weights = 1 * np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))

    return fn


def get_sample_fn(body, joints):
    def fn():
        values = []
        for joint in joints:
            limits = CIRCULAR_LIMITS if is_circular(body, joint) \
                else get_joint_limits(body, joint)
            values.append(np.random.uniform(*limits))
        return list(values)

    return fn


def get_extend_fn(body, joints, resolutions=None):
    if resolutions is None:
        resolutions = 0.05 * np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)

    def fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        num_steps = int(max(steps))
        waypoints = []
        diffs = difference_fn(q2, q1)
        for i in range(num_steps):
            waypoints.append(
                list(((float(i) + 1.0) / float(num_steps)) * np.array(diffs) + q1))
        return waypoints

    return fn


# Collision

def pairwise_collision(body1, body2, max_distance=MAX_DISTANCE):  # 10000
    # getContactPoints
    # return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0
    return p.getContactPoints(body1, body2) != ()


def pairwise_link_collision(body1, link1, body2, link2, max_distance=MAX_DISTANCE):  # 10000
    # return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
    #                               linkIndexA=link1, linkIndexB=link2)) != 0  # getContactPoints
    return p.getContactPoints(
        bodyA=body1,
        bodyB=body2,
        linkIndexA=link1,
        linkIndexB=link2)


def single_collision(body1, **kwargs):
    for body2 in get_bodies():
        if (body1 != body2) and pairwise_collision(body1, body2, **kwargs):
            return True
    return False


def all_collision(**kwargs):
    bodies = get_bodies()
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if pairwise_collision(bodies[i], bodies[j], **kwargs):
                return True
    return False


def get_moving_links(body, moving_joints):
    moving_links = list(moving_joints)
    for link in moving_joints:
        moving_links += get_link_descendants(body, link)
    return list(set(moving_links))


def get_moving_pairs(body, moving_joints):
    moving_links = get_moving_links(body, moving_joints)
    for i in range(len(moving_links)):
        link1 = moving_links[i]
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        for j in range(i + 1, len(moving_links)):
            link2 = moving_links[j]
            ancestors2 = set(get_joint_ancestors(
                body, link2)) & set(moving_joints)
            if ancestors1 != ancestors2:
                yield link1, link2


def get_self_link_pairs(body, joints, disabled_collisions=set()):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if True:
        check_link_pairs += list(get_moving_pairs(body, joints))
    else:
        check_link_pairs += list(combinations(moving_links, 2))
    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs


def get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions):
    check_link_pairs = get_self_link_pairs(
        body, joints, disabled_collisions) if self_collisions else []
    moving_bodies = [body] + [attachment.child for attachment in attachments]
    if obstacles is None:
        obstacles = list(set(get_bodies()) - set(moving_bodies))
    # + list(combinations(moving_bodies, 2))
    check_body_pairs = list(product(moving_bodies, obstacles))

    def collision_fn(q):
        if violates_limits(body, joints, q):
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        for link1, link2 in check_link_pairs:
            if pairwise_link_collision(body, link1, body, link2):
                return True
        return any(pairwise_collision(*pair) for pair in check_body_pairs)

    return collision_fn


def get_goal_test_fn(goal_conf, atol=0.001, rtol=0):
    def fn(conf):
        return np.allclose(conf, goal_conf, atol=atol, rtol=rtol)

    return fn


# Body and base

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])


# Bodies

def get_bodies():
    return [p.getBodyUniqueId(i)
            for i in range(p.getNumBodies())]


def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body))


def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')


def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')


def get_name(body):
    name = get_body_name(body)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))


def has_body(name):
    try:
        body_from_name(name)
    except ValueError:
        return False
    return True


def body_from_name(name):
    for body in get_bodies():
        if get_body_name(body) == name:
            return body
    raise ValueError(name)


def remove_body(body):
    return p.removeBody(body)


def get_body_pos(body):
    return get_body_pose(body)[0]


def get_body_quat(body):
    return get_body_pose(body)[1]  # [x,y,z,w]


def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat)


def set_point(body, point):
    set_pose(body, (point, get_body_quat(body)))


def set_quat(body, quat):
    set_pose(body, (get_body_pos(body), quat))


def is_rigid_body(body):
    for joint in get_joints(body):
        if is_movable(body, joint):
            return False
    return True


def is_fixed_base(body):
    return get_mass(body) == STATIC_MASS


def dump_body(body):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body), is_rigid_body(body), is_fixed_base(body)))
    for joint in get_joints(body):
        print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
            joint, get_joint_name(
                body, joint), JOINT_TYPES[get_joint_type(body, joint)],
            is_circular(body, joint), get_joint_limits(body, joint)))
    print('Link id: {} | Name: {} | Mass: {}'.format(-1,
                                                     get_base_name(body), get_mass(body)))
    for link in get_links(body):
        print('Link id: {} | Name: {} | Parent: {} | Mass: {}'.format(
            link, get_link_name(body, link), get_link_name(
                body, get_link_parent(body, link)),
            get_mass(body, link)))
        # print(get_joint_parent_frame(body, link))
        # print(map(get_data_geometry, get_visual_data(body, link)))
        # print(map(get_data_geometry, get_collision_data(body, link)))


def dump_world():
    for body in get_bodies():
        dump_body(body)
        print()


def remove_all_bodies():
    for i in get_body_ids():
        p.removeBody(i)


def reset_body_base(body, pose):
    p.resetBasePositionAndOrientation(body, pose[0], pose[1])


def get_body_infos():
    """ Return all body info in a list """
    return [get_body_info(i) for i in get_body_ids()]


def get_body_names():
    """ Return all body names in a list """
    return [bi.body_name for bi in get_body_infos()]


def get_body_id(name):
    return get_body_names().index(name)


def get_body_ids():
    return sorted([p.getBodyUniqueId(i) for i in range(p.getNumBodies())])


def get_body_pose(body):
    raw = p.getBasePositionAndOrientation(body)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]


# Control

def control_joint(body, joint, value):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint))


def control_joints(body, joints, positions, velocity=1.0, acceleration=2.0):
    return p.setJointMotorControlArray(body,
                                       joints,
                                       p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       positionGains=[velocity] * len(joints))


def forward_kinematics(body, joints=None, positions=None, eef_link=None):
    """
    Skip the resetJointState step,
    as it's not recommend to do this while running simulation.
    (https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#)
    """
    eef_link = get_num_joints(body) - 1 if eef_link is None else eef_link  # not true for dVRK models
    if joints is not None:
        # not recommend to resetJointState while running simulation
        old_positions = get_joint_positions(body, joints)
        set_joint_positions(body, joints, positions)
        eef_pose = get_link_pose(body, eef_link)
        set_joint_positions(body, joints, old_positions)
    else:
        eef_pose = get_link_pose(body, eef_link)
    return eef_pose


def inverse_kinematics(body, eef_link, position, orientation=None):
    """
    Modify the threshold and iteration parameters
    """
    if orientation is None:
        jv = p.calculateInverseKinematics(bodyUniqueId=body,
                                          endEffectorLinkIndex=eef_link,
                                          targetPosition=position,
                                          residualThreshold=1e-12,
                                          maxNumIterations=1000)
    else:
        jv = p.calculateInverseKinematics(bodyUniqueId=body,
                                          endEffectorLinkIndex=eef_link,
                                          targetPosition=position,
                                          targetOrientation=orientation,
                                          residualThreshold=1e-12,
                                          maxNumIterations=1000)
    return jv


# Links

BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints


def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')


def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex


def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_joints(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)


def has_link(body, name):
    try:
        link_from_name(body, name)
    except ValueError:
        return False
    return True


LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])


def get_link_state(body, link):
    return LinkState(*p.getLinkState(body, link))


def get_com_pose(body, link):  # COM = center of mass
    link_state = get_link_state(body, link)
    return list(link_state.linkWorldPosition), list(link_state.linkWorldOrientation)


def get_link_inertial_pose(body, link):
    link_state = get_link_state(body, link)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation


def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_body_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link)
    return [list(link_state.worldLinkFramePosition), list(link_state.worldLinkFrameOrientation)]


def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}


def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])


def get_link_ancestors(body, link):
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]


def get_joint_ancestors(body, link):
    return get_link_ancestors(body, link) + [link]


def get_movable_joint_ancestors(body, link):
    return list(filter(lambda j: is_movable(body, j), get_joint_ancestors(body, link)))


def get_link_descendants(body, link):
    descendants = []
    for child in get_link_children(body, link):
        descendants.append(child)
        descendants += get_link_descendants(body, child)
    return descendants


def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)


def get_adjacent_links(body):
    adjacent = set()
    for link in get_links(body):
        parent = get_link_parent(body, link)
        adjacent.add((link, parent))
        # adjacent.add((parent, link))
    return adjacent


def get_adjacent_fixed_links(body):
    return list(filter(lambda item: not is_movable(body, item[0]),
                       get_adjacent_links(body)))


def get_fixed_links(body):
    edges = defaultdict(list)
    for link, parent in get_adjacent_fixed_links(body):
        edges[link].append(parent)
        edges[parent].append(link)
    visited = set()
    fixed = set()
    for initial_link in get_links(body):
        if initial_link in visited:
            continue
        cluster = [initial_link]
        queue = deque([initial_link])
        visited.add(initial_link)
        while queue:
            for next_link in edges[queue.popleft()]:
                if next_link not in visited:
                    cluster.append(next_link)
                    queue.append(next_link)
                    visited.add(next_link)
        fixed.update(product(cluster, cluster))
    return fixed


DynamicsInfo = namedtuple('DynamicsInfo', ['mass', 'lateral_friction',
                                           'local_inertia_diagonal', 'local_inertial_pos', 'local_inertial_orn',
                                           'restitution', 'rolling_friction', 'spinning_friction',
                                           'contact_damping', 'contact_stiffness'])


def get_dynamics_info(body, link=BASE_LINK):
    return DynamicsInfo(*p.getDynamicsInfo(body, link))


def get_mass(body, link=BASE_LINK):
    return get_dynamics_info(body, link).mass


def get_joint_inertial_pose(body, joint):
    dynamics_info = get_dynamics_info(body, joint)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn


# Camera

CameraInfo = namedtuple('CameraInfo', ['width', 'height',
                                       'viewMatrix', 'projectionMatrix', 'cameraUp',
                                       'cameraForward', 'horizontal', 'vertical',
                                       'yaw', 'pitch', 'dist', 'target'])


def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
    p.resetDebugVisualizerCamera(
        cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)


def get_camera():
    return CameraInfo(*p.getDebugVisualizerCamera())


def render_image(width, height, view_matrix, proj_matrix, shadow=1):
    (_, _, px, _, mask) = p.getCameraImage(width=width,
                                           height=height,
                                           viewMatrix=view_matrix,
                                           projectionMatrix=proj_matrix,
                                           shadow=shadow,
                                           lightDirection=(10, 0, 10),
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array, mask
