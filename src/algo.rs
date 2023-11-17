use rustfft::num_complex::Complex;

pub const WINDOW_SIZE: usize = 1024;
pub const BLOCK_SIZE: usize = WINDOW_SIZE * 2;
pub const FFT_SIZE: usize = 8192;
pub const CMNDF_THRESHOLD: f32 = 0.03;
pub const A4_FREQUENCY: f32 = 440.0;

// Cumulative mean normalized difference function
pub fn compute_cmndf(x: &[f32]) -> Vec<f32> {
    let samples_count = x.len() / 2;
    let mut sum_difference_squared = vec![0.0; samples_count];
    for lag in 0..samples_count {
        let mut sum = 0.0;
        for i in 0..samples_count {
            let val = x[i] - x[i + lag];
            sum += val * val;
        }
        sum_difference_squared[lag] = sum;
    }
    let mut cmndf = vec![0.0; samples_count];
    cmndf[0] = 1.0;
    let mut prefix_sum = 0.0;
    for lag in 1..samples_count {
        prefix_sum += sum_difference_squared[lag];
        cmndf[lag] = sum_difference_squared[lag];
        if prefix_sum != 0.0 {
            cmndf[lag] /= prefix_sum / lag as f32;
        }
    }
    cmndf
}

// Hann window
pub fn apply_window(data: &mut [Complex<f32>]) {
    let fft_size = data.len() as f32;
    for (i, x) in data.iter_mut().enumerate() {
        *x *= Complex {
            re: 0.5 * (1.0 - f32::cos(2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1.0))),
            im: 0.0,
        };
    }
}

fn parabolic_interpolation(u: f32, v: f32, w: f32) -> f32 {
    0.5 * (u - w) / (u - 2.0 * v + w)
}

pub fn compute_f0(cmndf: &[f32], sample_rate: f32) -> f32 {
    let mut min_pos = 20;
    while cmndf[min_pos] > CMNDF_THRESHOLD && min_pos < WINDOW_SIZE - 1 {
        min_pos += 1;
    }

    if min_pos == WINDOW_SIZE - 1 {
        min_pos = cmndf[1..WINDOW_SIZE - 1]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap()
            + 1;
    }

    sample_rate
        / (min_pos as f32
            + parabolic_interpolation(cmndf[min_pos - 1], cmndf[min_pos], cmndf[min_pos + 1]))
}

pub fn lerp(a: f32, b: f32, alpha: f32) -> f32 {
    a + alpha * (b - a)
}

pub fn get_cents(frequency: f32, target_frequency: f32) -> f32 {
    f32::log2(frequency / target_frequency) * 12.0 * 100.0
}

// note index is the number of semitones between A4 and the note
fn get_note_index(frequency: f32) -> isize {
    let semitones = f32::log2(frequency / A4_FREQUENCY) * 12.0;
    semitones.round() as isize
}

pub fn get_note(frequency: f32) -> (isize, f32) {
    let note = get_note_index(frequency);
    let exact_frequency = A4_FREQUENCY * compute_interval_ratio(note);
    (note, exact_frequency)
}

pub fn compute_interval_ratio(semitones: isize) -> f32 {
    2.0_f32.powf(semitones as f32 / 12.0)
}
