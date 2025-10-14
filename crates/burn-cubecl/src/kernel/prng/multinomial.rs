use burn_tensor::Shape;
use cubecl::{prelude::*, std::tensor::ViewOperations};
use num_traits::PrimInt;

use crate::{
    CubeElement, CubeRuntime,
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    ops::numeric::empty_device,
    tensor::CubeTensor,
};

use super::{N_VALUES_PER_THREAD, get_seeds, prng_cube_count};

/// Pseudo-random generator
fn random<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    args: TensorArg<'_, R>,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device::<R, E>(client.clone(), device.clone(), shape);
    let seeds = get_seeds();

    let cube_dim = CubeDim::default();
    let cube_count = prng_cube_count(output.shape.num_elements(), cube_dim, N_VALUES_PER_THREAD);

    prng_kernel::launch::<E, R>(
        &client,
        cube_count,
        cube_dim,
        output.as_tensor_arg::<E>(1),
        ScalarArg::new(seeds[0]),
        ScalarArg::new(seeds[1]),
        ScalarArg::new(seeds[2]),
        ScalarArg::new(seeds[3]),
        args,
        N_VALUES_PER_THREAD as u32,
    );

    output
}

#[cube(launch)]
fn prng_kernel<E: CubeElement>(
    output: &mut Tensor<E>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: Tensor<E>,
    #[comptime] n_values_per_thread: u32,
) {
    let cube_offset = CUBE_POS * CUBE_DIM;

    let write_index_base = cube_offset * n_values_per_thread + UNIT_POS;

    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * ABSOLUTE_POS;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;
    let n_invocations = CUBE_DIM;

    // Creation of n_values_per_thread values, specific to the distribution
    let prob = 23.0; // TODO
    let should_unroll = n_values_per_thread <= 8;

    #[unroll(should_unroll)]
    for i in 0..n_values_per_thread {
        state_0 = taus_step_0(state_0);
        state_1 = taus_step_1(state_1);
        state_2 = taus_step_2(state_2);
        state_3 = lcg_step(state_3);

        let int_random = state_0 ^ state_1 ^ state_2 ^ state_3;
        let float_random = cast_uint_to_float(int_random);
        let write_index = i * n_invocations + write_index_base;

        output[write_index] = E::cast_from(float_random < prob);
    }

    // let seeds = at::cuda::philox::unpack(philox_args);

    // global index formula for 2D grid of 1D blocks
    let idx = CUBE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X + CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;

    let distributions = 1; // DEV: We always flattent the input tensor, so it is treated as one flat distribution
    let args_flat = args.flatten().collect();
    let categories = args_flat.len();
    let totalSamples = todo!("Total number of elements in the output Tensor");

    let max_val = args_flat.max();
    let min_val = args_flat.min();

    let normDist = (args_flat - min_val) / max_val;
    let normDistPrefixSum = normDist.fold(0, |state, val| state += val);

    // curandStatePhilox4_32_10_t state;
    // curand_init(std::get<0>(seeds),
    //             idx,
    //             std::get<1>(seeds),
    //             &state);

    // The block determines the distribution for which we generate a point
    // for (int64_t curDist = CUBE_POS_Y;
    //      curDist < distributions;
    //      curDist += CUBE_COUNT_Y) {
    for curr_dist in (CUBE_POS_Y..distributions).step_by(CUBE_COUNT_Y) {
        // for (int sample = CUBE_POS_X*CUBE_DIM_X + UNIT_POS_X;
        //      sample < totalSamples; sample += CUBE_DIM_X*CUBE_COUNT_X) {
        for sample in
            (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X..totalSamples).step_by(CUBE_DIM_X * CUBE_COUNT_X)
        {
            //we are losing 3 out of 4 generated numbers but it's ok
            //this kernel is not very efficient anyway
            // auto rand = curand_uniform4(&state);
            // scalar_t r = static_cast<scalar_t>(rand.x);
            state_0 = taus_step_0(state_0);
            state_1 = taus_step_1(state_1);
            state_2 = taus_step_2(state_2);
            state_3 = lcg_step(state_3);

            let int_random = state_0 ^ state_1 ^ state_2 ^ state_3;
            let elem_random = E::from_int(int_random);

            // Find the bucket that a uniform sample lies in
            let choice = binary_search_for_multinomial(
                normDistPrefixSum + curr_dist * categories,
                normDist + curr_dist * categories,
                categories,
                elem_random,
            );

            output[curr_dist * totalSamples + sample] = choice;
        }
    }
}

#[cube]
fn binary_search_for_multinomial<E: CubeElement>(
    cumdist: &Tensor<E>,
    dist: &Tensor<E>,
    size: u32,
    val: E,
) -> u32 {
    let mut start = 0;
    let mut end = size;
    // cumdist[size - 1] = 0 => all zero prob dist
    // TODO: PyTorch checks that all values in cumdist are greater than 0 here
    // CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));

    while (end - start > 0) {
        let mid = start + (end - start) / 2;

        let mid_val: E = cumdist[mid];
        if (mid_val < val) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    if (start == size) {
        // No probability mass or precision problems; just return the
        // first non-zero element by setting start to size-1 here,
        // the code below will move it to the last non-zero probability
        // this actually can happen when the random number is 1
        // (github pytorch issue #4858).
        start = size - 1;
    }

    while (start >= 1 && dist[start] == 0) {
        start -= 1;
    }

    start
}

#[derive(CubeLaunch)]
struct Multinomial<E: Numeric> {
    probabilities: Tensor<E>,
}

/// Pseudo-random generator with uniform distribution
pub fn random_multinomial<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    props: CubeTensor<R>,
) -> CubeTensor<R> {
    random::<R, E>(shape, device, props.as_tensor_arg::<E>(1))
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_multinomial<R: CubeRuntime, E: CubeElement>(
    tensor: &CubeTensor<R>,
    props: CubeTensor<R>,
) -> CubeTensor<R> {
    random_multinomial::<R, E>(tensor.shape.clone(), &tensor.device, props)
}
