import tensorflow.experimental.numpy as tnp  # type: ignore


def apply_transform(vector, transform):
    """This function handles the transformation of 2D vectors.

    Keypoints come as [[x1, y1], [x2, y2], ...] or batched as
    [[[x1, y1], [x2, y2], ...], [[x3, y3], [x4, y4], ...], ...]

    Args:
        vector (tf.Tensor): Keypoint vector, either Nx2 or BxNx2 (N keypoints, B batches)
        transform (tf.Tensor): Transformation matrix (3x3) or
                               batch of transformation matrices (Bx3x3)

    Returns:
        tf.Tensor: Transformed keypoints
    """
    shape = vector.shape

    if shape[-1] != 2:
        raise ValueError(f"Last dimension must be 2 but {shape[-1]} was given "
                         f"(from shape {shape}).")

    if len(shape) == 3:
        N, C, _ = shape
        # combine batches and channels
        vector = tnp.reshape(vector, (N*C, 2))
    elif len(shape) == 2:
        N = None
        C, _ = shape
    else:
        raise ValueError(f"Invalid vector shape: {shape}")

    homogeneous_vector = tnp.stack((vector[:, 0], vector[:, 1], tnp.ones_like(vector[:, 0])))

    if len(transform.shape) == 3:
        transform = tnp.repeat(transform, C, axis=0)
        result = tnp.einsum('in,nji->nj', homogeneous_vector, transform)
    elif len(transform.shape) == 2:
        result = tnp.einsum('in, ji->nj', homogeneous_vector, transform)
    elif len(transform.shape) != 2:
        raise ValueError(f"Invalid transform shape: {transform.shape}")

    result = result[:, :2]

    # reshape back into batches
    if N is not None:
        result = tnp.reshape(result, (N, C, 2))

    return result


def scaling_transform(scaling):
    m_scale = tnp.array([
        [scaling, 0, 0],
        [0, scaling, 0],
        [0, 0, 1]
    ])

    return m_scale


def translation_transform(translation_x, translation_y):
    m_trans = tnp.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ])

    return m_trans


def newell_cropping_transform(center, body_scale, crop_resolution):
    body_size = 200 * body_scale  # from mpii dataset README.md file
    top_left = center - body_size/2

    m_trans = translation_transform(-top_left[0], -top_left[1])
    m_scale = scaling_transform(crop_resolution / body_size)

    return m_scale @ m_trans


def rotation_transform(angle, resolution):
    angle_rad = angle / 180 * tnp.pi
    cos = tnp.cos(angle_rad)
    sin = tnp.sin(angle_rad)

    m_rot = tnp.array([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [0, 0, 1]
    ])

    m_trans = translation_transform(resolution/2, resolution/2)
    m_trans_inv = translation_transform(-resolution/2, -resolution/2)

    # Move point to center, then rotate, then move back
    return m_trans @ m_rot @ m_trans_inv


def horizontal_flip_transform(resolution):
    m_trans = translation_transform(-(resolution-1) / 2, 0)

    m_flip = tnp.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    m_trans_inv = translation_transform((resolution-1) / 2, 0)
    return m_trans_inv @ m_flip @ m_trans


# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def permutation_transform(after: list):
    """Permutation transform.

    For example if permute = tnp.array([2, 0, 1]) then this will return
    tnp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    This is not compatible with apply_transform.
    Simply use matrix multiplication operator:
    ```
    result = permute_t @ keypoints
    ```
    """
    return tnp.eye(len(after))[tnp.array(after)]
