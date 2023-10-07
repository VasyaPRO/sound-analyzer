use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind, KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    prelude::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    widgets::{
        Axis, Bar, BarChart, BarGroup, Block, BorderType, Borders, Chart, Dataset, GraphType,
        Paragraph,
    },
    Terminal,
};
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};
use std::{io, sync::atomic::AtomicUsize, time::Duration};

use crate::ascii_symbols::get_note_ascii;

const WINDOW_SIZE: usize = 1024;

struct State {
    running: bool,
    apply_window: bool,
    target_frequency: Option<f32>,
}

impl Default for State {
    fn default() -> Self {
        State {
            running: true,
            apply_window: true,
            target_frequency: None,
        }
    }
}

fn parabolic_interpolation(u: f32, v: f32, w: f32) -> f32 {
    0.5 * (u - w) / (u - 2.0 * v + w)
}

mod ascii_symbols {

pub const ASCII_LETTERS: [[&str; 7]; 17] = [
    [
        "  ## ##   ",
        "  ## ##   ",
        "######### ",
        " ## ##    ",
        "######### ",
        "  ## ##   ",
        "  ## ##   ",
    ],
    [
        "   ##   ",
        " ####   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        " ###### ",
    ],
    [
        " #######  ",
        "##     ## ",
        "       ## ",
        " #######  ",
        "##        ",
        "##        ",
        "######### ",
    ],
    [
        " #######  ",
        "##     ## ",
        "       ## ",
        " #######  ",
        "       ## ",
        "##     ## ",
        " #######  ",
    ],
    [
        "##    ##  ",
        "##    ##  ",
        "##    ##  ",
        "##    ##  ",
        "######### ",
        "      ##  ",
        "      ##  ",
    ],
    [
        "########  ",
        "##        ",
        "##        ",
        "#######   ",
        "      ##  ",
        "##    ##  ",
        " ######   ",
    ],
    [
        " #######  ",
        "##     ## ",
        "##        ",
        "########  ",
        "##     ## ",
        "##     ## ",
        " #######  ",
    ],
    [
        "########  ",
        "##    ##  ",
        "    ##    ",
        "   ##     ",
        "  ##      ",
        "  ##      ",
        "  ##      ",
    ],
    [
        " #######  ",
        "##     ## ",
        "##     ## ",
        " #######  ",
        "##     ## ",
        "##     ## ",
        " #######  ",
    ],
    [
        " #######  ",
        "##     ## ",
        "##     ## ",
        " ######## ",
        "       ## ",
        "##     ## ",
        " #######  ",
    ],
    [
        "   ###    ",
        "  ## ##   ",
        " ##   ##  ",
        "##     ## ",
        "######### ",
        "##     ## ",
        "##     ## ",
    ],
    [
        "########  ",
        "##     ## ",
        "##     ## ",
        "########  ",
        "##     ## ",
        "##     ## ",
        "########  ",
    ],
    [
        " ######  ",
        "##    ## ",
        "##       ",
        "##       ",
        "##       ",
        "##    ## ",
        " ######  ",
    ],
    [
        "########  ",
        "##     ## ",
        "##     ## ",
        "##     ## ",
        "##     ## ",
        "##     ## ",
        "########  ",
    ],
    [
        "######## ",
        "##       ",
        "##       ",
        "######   ",
        "##       ",
        "##       ",
        "######## ",
    ],
    [
        "######## ",
        "##       ",
        "##       ",
        "######   ",
        "##       ",
        "##       ",
        "##       ",
    ],
    [
        " ######  ",
        "##    ## ",
        "##       ",
        "##   ### ",
        "##    ## ",
        "##    ## ",
        " ######  ",
    ],
];


pub fn get_note_ascii(note: i32) -> Vec<String> {
    let octave = (note.div_euclid(12) + 4).clamp(1, 9) as usize;

    fn combine_two(a: usize, b: usize) -> Vec<String> {
        ASCII_LETTERS[a]
            .iter()
            .zip(ASCII_LETTERS[b])
            .map(|(&s, t)| s.to_owned() + t)
            .collect()
    }

    fn combine_three(a: usize, b: usize, c: usize) -> Vec<String> {
        ASCII_LETTERS[a]
            .iter()
            .zip(ASCII_LETTERS[b])
            .zip(ASCII_LETTERS[c])
            .map(|((&s, t), u)| s.to_owned() + t + u)
            .collect()
    }

    match note.rem_euclid(12) {
        0 => combine_two(10, octave),
        1 => combine_three(10, 0, octave),
        2 => combine_two(11, octave),
        3 => combine_two(12, octave),
        4 => combine_three(12, 0, octave),
        5 => combine_two(13, octave),
        6 => combine_three(13, 0, octave),
        7 => combine_two(14, octave),
        8 => combine_two(15, octave),
        9 => combine_three(15, 0, octave),
        10 => combine_two(16, octave),
        11 => combine_three(16, 0, octave),
        _ => unreachable!()
    }
}

}
// use ascii_symbols::ASCII_LETTERS;

// Cumulative mean normalized difference function
fn compute_cmndf(x: &[f32]) -> Vec<f32> {
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
        let den = if prefix_sum == 0.0 {
            1.0
        } else {
            prefix_sum / lag as f32
        };
        cmndf[lag] = sum_difference_squared[lag] / den;
    }
    cmndf
}

fn get_cents(frequency: f32, target_frequency: f32) -> f32 {
    f32::log2(frequency / target_frequency) * 12.0 * 100.0
}

const A4_FREQUENCY: f32 = 440.0;

fn get_note_index(frequency: f32) -> i32 {
    let semitones = f32::log2(frequency / A4_FREQUENCY) * 12.0;
    semitones.round() as i32
}

fn get_note(frequency: f32) -> (i32, f32) {
    // const A4_FREQUENCY: f32 = 440.0;
    // let semitones = f32::log2(frequency / A4_FREQUENCY) * 12.0;
    // const NOTES: [&str; 12] = [
    //     "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
    // ];
    // let note = semitones.round() as i32;
    let note = get_note_index(frequency);
    // let octave = note.div_euclid(12) + 4;
    let exact_frequency = A4_FREQUENCY * 2.0_f32.powf(note as f32 / 12.0);
    // let cents = f32::log2(frequency / exact_frequency) * 12.0 * 100.0;
    // freq / exact = 2^(0.5 / 12)
    // log2(freq / exact) = 0.5 / 12
    // 0.5 = 12 * log(freq / exact)
    (
        // NOTES[note.rem_euclid(12) as usize].to_owned() + &octave.to_string(),
        note,
        // get_cents(frequency, exact_frequency),
        exact_frequency,
    )
}

// fn get_note_ascii(note: i32) -> Vec<String> {

// Hann window
fn apply_window(data: &mut [Complex<f32>]) {
    let fft_size = data.len() as f32;
    for (i, x) in data.iter_mut().enumerate() {
        *x *= Complex {
            re: 0.5 * (1.0 - f32::cos(2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1.0))),
            im: 0.0,
        };
    }
}

fn main() -> io::Result<()> {
    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let host = cpal::default_host();
    // if let Ok(input_devices) = host.input_devices() {
    //     for device in input_devices {
    //         if let Ok(c) = device.supported_input_configs() {
    //             for x in c {
    //                 // dbg!(x);
    //             }
    //         }
    //     }
    // }
    let device = host.default_input_device().unwrap();
    let mut supported_configs_range = device
        .supported_input_configs()
        .expect("error while querying configs");
    // println!("Input device: {:?}", device.name());
    let supported_config = supported_configs_range
        .next()
        .expect("no supported config?!")
        // .with_sample_rate(cpal::SampleRate(48000));
        .with_max_sample_rate();
    let config = supported_config.config();
    // println!("hello");
    // dbg!(device);
    // let cnt = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // let cnt2 = cnt.clone();
    let (sender, reciever) = std::sync::mpsc::channel::<f32>();
    let atomic_cnt = AtomicUsize::new(0);
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // sample from two channels are received interleaved
                // we only take one channel now
                for i in (0..data.len()).step_by(2) {
                    // sender.send(data[i]).unwrap();
                    let val = f32::sin(
                        atomic_cnt.load(std::sync::atomic::Ordering::SeqCst) as f32
                            / config.sample_rate.0 as f32
                            * f32::acos(-1.0)
                            * 2.0
                            * 48000.0
                            / 24.0,
                    );
                    atomic_cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    sender.send(data[i]).unwrap();
                    // sender.send(val).unwrap();
                }
            },
            move |_err| {
                // dbg!(err);
                // println!("err");
            },
            None, // None=blocking, Some(Duration)=timeout
        )
        .unwrap();
    stream.play().unwrap();
    // stream.
    // std::thread::sleep(std::time::Duration::new(3, 0));
    // drop(stream);
    let mut samples = Vec::new();
    const BLOCK_SIZE: usize = WINDOW_SIZE * 2;
    const FFT_SIZE: usize = 8192;
    let mut min_amount = BLOCK_SIZE.max(FFT_SIZE);
    let mut frequencies = Vec::new();

    // let mut running = true;
    let mut state = State::default();
    let mut avg_f0 = 0.0;
    let mut avg_db = 0.0;
    // let cursor_pos = backend.get_cursor()

    let mut cursor_pos = (0, 0);
    while state.running {
        while let Ok(sample) = reciever.try_recv() {
            assert!(f32::abs(sample) <= 1.0);
            samples.push(sample);
        }

        if samples.len() >= min_amount {
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);

            let mut fft_buf = samples[min_amount - FFT_SIZE..min_amount]
                .iter()
                .map(|x| Complex { re: *x, im: 0.0 })
                .collect::<Vec<_>>();
            // let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; BLOCK_SIZE];

            if state.apply_window {
                apply_window(&mut fft_buf);
            }
            fft.process(&mut fft_buf);

            // let freq: f32 = 0.0;
            // dbg!(min_amount - BLOCK_SIZE, min_amount, samples.len());
            let cmndf = compute_cmndf(&samples[min_amount - BLOCK_SIZE..min_amount]);
            let rms = f32::sqrt(
                samples[min_amount - BLOCK_SIZE..min_amount]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    / BLOCK_SIZE as f32,
            );
            // frequencies.push(freq);
            let mut min_pos = 1;
            while cmndf[min_pos] > 0.03 && min_pos < WINDOW_SIZE - 1 {
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
            // let sample_rate = 48000;
            // let sr = config.sample_rate;
            let f0: f32 = config.sample_rate.0 as f32
                / (min_pos as f32
                    + parabolic_interpolation(
                        cmndf[min_pos - 1],
                        cmndf[min_pos],
                        cmndf[min_pos + 1],
                    ));
            avg_f0 = avg_f0 + 0.1 * (f0 - avg_f0);
            // avg_f0 = f0;
            // println!("{}hz", f0);
            // println!("{}rms mn_pos{}", rms, min_pos);
            // println!("{}db", 20.0 * f32::log10(rms));
            let (note_index, cents) = match state.target_frequency {
                Some(target_frequency) => {
                    // let (note, exact_frequency) = get_note(avg_f0);
                    let cents = get_cents(avg_f0, target_frequency);
                    let (target_note, _) = get_note(target_frequency);
                    (target_note, cents)
                }
                None => {
                    let (note, exact_frequency) = get_note(avg_f0);
                    let cents = get_cents(avg_f0, exact_frequency);
                    (note, cents)
                }
            };



            // println!("{} {:+}", note, cents);
            frequencies.push(f0);
            min_amount += BLOCK_SIZE / 4;

            //
            // let cursor_pos = layout.
            //
            let terminal_width = terminal.size().unwrap().width;
            terminal.draw(|f| {
                // let layout = Layout::default()
                //     // .direction(Direction::Vertical)
                //     .direction(Direction::Vertical)
                //     .constraints(vec![Constraint::Length(5), Constraint::Min(0)])
                //     .split(Rect::new(0, 0, 10, 10));
                // .split(Rect::new(0, 0, 10, 10));
                let layout = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(1)
                    .constraints(
                        [
                            Constraint::Min(7),
                            Constraint::Percentage(20),
                            Constraint::Percentage(60),
                            Constraint::Percentage(10),
                        ]
                        .as_ref(),
                    )
                    .split(f.size());

                let db = 20.0 * f32::log10(rms);

                // avg_db = avg_db + 0.1 * (db - avg_db);
                avg_db = db;


                f.render_widget(
                    Paragraph::new(get_note_ascii(note_index).join("\n")),
                    layout[0]);

                // f.render_widget(
                //     Paragraph::new(get_note_ascii(get_note_index(avg_f0)).join("\n")),
                    // .block(
                    //     Block::default()
                    //         // .title("Template")
                    //         // .title_alignment(Alignment::Center)
                    //         // .borders(Borders::ALL)
                    //         // .border_type(BorderType::Rounded),
                    // )
                    // .style(Style::default().fg(Color::Cyan).bg(Color::Black))
                    // .style(Style::default().fg(Color::Cyan))
                    // .alignment(Alignment::Center),
                    // Paragraph::new(format!(
                    //     "Frequency: {:9.3}\n\
                    // Note: {:3} {:+2.3}\n\
                    // {:.3} db\n\
                    // Press `Esc`, `Ctrl-C` or `q` to stop running.",
                    //     avg_f0, note, cents, db
                    // ))
                    // .block(
                    //     Block::default()
                    //         .title("Template")
                    //         .title_alignment(Alignment::Center)
                    //         .borders(Borders::ALL)
                    //         .border_type(BorderType::Rounded),
                    // )
                    // .style(Style::default().fg(Color::Cyan).bg(Color::Black))
                    // .alignment(Alignment::Center),
                //     layout[0],
                // );

                let db_bar = Bar::default()
                    .value((db + 100.0) as u64)
                    .text_value(format!("{:5.01}db", db));

                let cents_bar = Bar::default()
                    .value((cents + 50.0) as u64)
                    .text_value(format!("{:+6.02}c", cents));

                let frequency_bar = Bar::default()
                    .value((f0 / 10.0) as u64)
                    .text_value(format!("{:6.02}hz", avg_f0));

                let bar_chart = BarChart::default()
                    // .block(Block::default().title("BarChart").borders(Borders::ALL))
                    .block(Block::default())
                    .direction(Direction::Horizontal)
                    // .bar_width(3)
                    // .bar_gap(1)
                    // .bar_style(Style::default().fg(Color::Yellow).bg(Color::Red))
                    // .value_style(Style::default().add_modifier(Modifier::HIDDEN))
                    // .label_style(Style::default().add_modifier(Modifier::HIDDEN))
                    // .bar_style(Style::default().add_modifier(Modifier::HIDDEN))
                    // .label_style(Style::default().fg(Color::White))
                    // .data(&[("db", (db + 100.0) as u64)])
                    .data(BarGroup::default().bars(&[db_bar, cents_bar, frequency_bar]))
                    .max(100);
                // let bg = BarGroup::default()
                // let bg = BarGroup::default()
                //     .label("Group 1".into())
                //     .bars(&[Bar::default().value(200), Bar::default().value(150)]);
                f.render_widget(bar_chart, layout[1]);

                let bin_step = config.sample_rate.0 as f32 / FFT_SIZE as f32;
                let spectrum_data = fft_buf[1..FFT_SIZE / 2 + 1]
                    .iter()
                    .enumerate()
                    .map(|p| {
                        (
                            f64::ln((bin_step * p.0 as f32).into()),
                            // p.1.abs() as f64 / FFT_SIZE as f64,
                            20.0 * f64::log10(p.1.abs() as f64 / FFT_SIZE as f64),
                            // let db = 20.0 * f32::log10(rms);
                        )
                    })
                    .collect::<Vec<_>>();
                // .collect::<Vec<_>>();
                let current_freq = [(f64::ln(state.target_frequency.unwrap_or(avg_f0).into()), -120.0),
                    (f64::ln(state.target_frequency.unwrap_or(avg_f0).into()), 0.0)];

                let ft_dataset = vec![Dataset::default()
                    // .name("Fourier transform")
                    //     .marker(symbols::Marker::Dot)
                    //     .graph_type(GraphType::Scatter)
                    //     .style(Style::default().fg(Color::Cyan))
                    //     .data(&[(0.0, 5.0), (1.0, 6.0), (1.5, 6.434)]),
                    // Dataset::default()
                    // .name("data2")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Magenta))
                    // .data(&[(4.0, 5.0), (5.0, 8.0), (7.66, 13.5)]),
                    .data(&spectrum_data),

                    Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Red))
                    .data(&current_freq),
                ];

                let ft_chart = Chart::new(ft_dataset)
                    // .block(Block::default().title("Chart"))
                    .x_axis(
                        Axis::default()
                            // .title(Span::styled("X Axis", Style::default().fg(Color::Red)))
                            .style(Style::default().fg(Color::White))
                            // .bounds([0.0, BLOCK_SIZE as f64 / 2.0])
                            .bounds([
                                f64::ln(bin_step.into()),
                                f64::ln(config.sample_rate.0 as f64 / 2.0),
                            ]),
                    )
                    .y_axis(
                        Axis::default()
                            // .title(Span::styled("Y Axis", Style::default().fg(Color::Red)))
                            .style(Style::default().fg(Color::White))
                            // .bounds([0.0, 0.5])
                            .bounds([-120.0, 0.0]),
                    );
                // ft_chart.layout();

                f.render_widget(ft_chart, layout[2]);

                let mut mx: f64 = 0.0;
                for x in &spectrum_data {
                    mx = mx.max(x.1);
                }

                let inside = cursor_pos.0 >= layout[2].top()
                    && cursor_pos.0 < layout[2].bottom()
                    && cursor_pos.1 > 0
                    && cursor_pos.1 < terminal_width - 1;
                let mut cursor_freq = 0.0;
                if inside {
                    // frequency = cursor_pos
                    // 1 -> log(bin)
                    // terminal_width - 1 -> log(sample_rate)
                    // terminal_width - 2

                    let log_x = f32::ln(bin_step)
                        + (cursor_pos.1 - 1) as f32
                            * f32::ln(config.sample_rate.0 as f32 / 2.0 / bin_step)
                            / (terminal_width - 3) as f32;
                    cursor_freq = f32::exp(log_x);
                    // cursor_pos.1 - 1
                    //01234
                    // ln(bin_step * p.0 as f64)
                    // ln
                }

                f.render_widget(
                    Paragraph::new(format!(
                        "sec: {:9.3}\n\
                        {}, {}, {bin_step} {inside}, {}
                    ",
                        mx, cursor_pos.0, cursor_pos.1, cursor_freq
                    ))
                    .block(
                        Block::default()
                            .title("Template")
                            .title_alignment(Alignment::Center)
                            .borders(Borders::ALL)
                            .border_type(BorderType::Rounded),
                    )
                    .style(Style::default().fg(Color::Cyan).bg(Color::Black))
                    .alignment(Alignment::Center),
                    layout[3],
                );
            })?;

            loop {
                match event::poll(Duration::from_millis(0)) {
                    Ok(ready) if ready => {
                        let ev = event::read().unwrap();
                        match ev {
                            // Event::Key(key) if key.modifiers == KeyModifiers::CONTROL => {
                            //     match key.code {
                            //         event::KeyCode::Char('c' | 'C') => {
                            //             state.running = false;
                            //         }
                            //         _ => {}
                            //     }
                            // }
                            Event::Key(key) => match key.code {
                                event::KeyCode::Char('c' | 'C')
                                    if key.modifiers == KeyModifiers::CONTROL =>
                                {
                                    state.running = false;
                                }
                                event::KeyCode::Char('w') if key.kind == KeyEventKind::Press => {
                                    state.apply_window = !state.apply_window;
                                }
                                event::KeyCode::Char('a') if key.kind == KeyEventKind::Press => {
                                    state.target_frequency = Some(440.0);
                                }
                                event::KeyCode::Up if key.kind == KeyEventKind::Press => {
                                    if let Some(freq) = state.target_frequency {
                                        state.target_frequency = Some(freq * 2.0);
                                    }
                                }
                                event::KeyCode::Down if key.kind == KeyEventKind::Press => {
                                    if let Some(freq) = state.target_frequency {
                                        state.target_frequency = Some(freq * 0.5);
                                    }
                                }
                                event::KeyCode::Right if key.kind == KeyEventKind::Press => {
                                    if let Some(freq) = state.target_frequency {
                                        state.target_frequency = Some(freq * 2.0.powf(1.0 / 12.0));
                                    }
                                }
                                event::KeyCode::Left if key.kind == KeyEventKind::Press => {
                                    if let Some(freq) = state.target_frequency {
                                        state.target_frequency = Some(freq * 2.0.powf(-1.0 / 12.0));
                                    }
                                }
                                event::KeyCode::Esc if key.kind == KeyEventKind::Press => {
                                    state.target_frequency = None;
                                }
                                _ => {}
                            },
                            Event::Mouse(e) if e.kind == MouseEventKind::Moved => {
                                cursor_pos = (e.row, e.column)
                            }
                            _ => {}
                        }
                    }
                    _ => break,
                }
            }
        }
    }

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}
