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
    prelude::{Backend, Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols,
    widgets::{Axis, Bar, BarChart, BarGroup, Block, Chart, Dataset, GraphType, Paragraph},
    Frame, Terminal,
};
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};
use std::{io, time::Duration};

use crate::{
    algo::{apply_window, compute_cmndf},
    ascii_symbols::get_note_ascii,
};

use self::algo::{
    compute_f0, compute_interval_ratio, get_cents, get_note, lerp, A4_FREQUENCY, BLOCK_SIZE,
    CMNDF_THRESHOLD, FFT_SIZE,
};

enum Graph {
    AmplitudeSpectrum,
    CMNDF,
}

struct State {
    running: bool,
    cursor_pos: (u16, u16),
    mouse_pressed: bool,
    sample_rate: f32,
    apply_window: bool,
    target_frequency: Option<f32>,
    graph: Graph,
}

impl State {
    fn with_sample_rate(sample_rate: f32) -> Self {
        State {
            running: true,
            apply_window: true,
            target_frequency: None,
            graph: Graph::AmplitudeSpectrum,
            cursor_pos: (0, 0),
            sample_rate,
            mouse_pressed: false,
        }
    }
}

#[rustfmt::skip]
mod ascii_symbols;
mod algo;

fn draw_graph<B>(
    frame: &mut Frame<B>,
    area: Rect,
    state: &State,
    fft_buf: &[Complex<f32>],
    cmndf: &[f32],
    current_frequency: f32,
) where
    B: Backend,
{
    let mut datasets = Vec::new();
    let graph_data;
    let x_bounds;
    let y_bounds;
    let cmndf_threshold;

    match state.graph {
        Graph::AmplitudeSpectrum => {
            graph_data = fft_buf[1..FFT_SIZE / 2 + 1]
                .iter()
                .enumerate()
                .map(|(index, &val)| {
                    (
                        f64::ln(((state.sample_rate / FFT_SIZE as f32) * index as f32).into()),
                        20.0 * f64::log10(f64::from(val.abs()) / FFT_SIZE as f64),
                    )
                })
                .collect::<Vec<_>>();

            x_bounds = [
                f64::ln((state.sample_rate / FFT_SIZE as f32).into()),
                f64::ln(state.sample_rate as f64 / 2.0),
            ];

            y_bounds = [-140.0, 0.0];
        }
        Graph::CMNDF => {
            graph_data = cmndf[1..]
                .iter()
                .enumerate()
                .map(|(index, &val)| {
                    (
                        f64::ln((state.sample_rate / (index + 1) as f32).into()),
                        f64::from(val),
                    )
                })
                .collect::<Vec<_>>();

            x_bounds = [
                f64::ln(state.sample_rate as f64 / (cmndf.len() - 1) as f64),
                f64::ln(state.sample_rate as f64 / 2.0),
            ];

            y_bounds = [0.0, 5.0];

            cmndf_threshold = [
                (x_bounds[0], CMNDF_THRESHOLD as f64),
                (x_bounds[1], CMNDF_THRESHOLD as f64),
            ];

            datasets.push(
                Dataset::default()
                    .name("CMNDF threshold")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Yellow))
                    .data(&cmndf_threshold),
            );
        }
    }

    let target_freq;
    if let Some(target_frequency) = state.target_frequency {
        target_freq = [
            (f64::ln(target_frequency.into()), y_bounds[0]),
            (f64::ln(target_frequency.into()), y_bounds[1]),
        ];

        datasets.push(
            Dataset::default()
                .name("Target frequency")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Gray))
                .data(&target_freq),
        );
    }

    let current_freq = [
        (f64::ln(current_frequency.into()), y_bounds[0]),
        (f64::ln(current_frequency.into()), y_bounds[1]),
    ];

    datasets.push(
        Dataset::default()
            .name("Current frequency")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(&current_freq),
    );

    datasets.push(
        Dataset::default()
            .name(match state.graph {
                Graph::AmplitudeSpectrum => "Amplitude spectrum",
                Graph::CMNDF => "Cumulative mean normalized difference function",
            })
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&graph_data),
    );

    let graph_chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::White))
                .bounds(x_bounds),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(Color::White))
                .bounds(y_bounds),
        )
        .hidden_legend_constraints((Constraint::Ratio(3, 5), Constraint::Ratio(1, 2)))
        .bg(Color::Black);

    frame.render_widget(graph_chart, area);
}

fn draw_bar_chart<B>(frame: &mut Frame<B>, area: Rect, db: f32, cents: f32)
where
    B: Backend,
{
    let db_bar = Bar::default()
        .value((db + 100.0) as u64)
        .text_value(format!("{:5.01}db", db));

    let cents_bar = Bar::default()
        .value((cents + 50.0) as u64)
        .text_value(format!("{:+6.02}c", cents));

    let bar_chart = BarChart::default()
        .block(Block::default())
        .direction(Direction::Horizontal)
        .data(BarGroup::default().bars(&[db_bar, cents_bar]))
        .max(100);

    frame.render_widget(bar_chart, area);
}

fn handle_events(state: &mut State) {
    loop {
        match event::poll(Duration::from_millis(0)) {
            Ok(ready) if ready => {
                let ev = event::read().unwrap();
                match ev {
                    Event::Key(key) => match key.code {
                        event::KeyCode::Char('c' | 'C')
                            if key.modifiers == KeyModifiers::CONTROL =>
                        {
                            state.running = false;
                        }
                        event::KeyCode::Char('w') if key.kind == KeyEventKind::Press => {
                            state.apply_window = !state.apply_window;
                        }
                        event::KeyCode::Char(note_char)
                            if ('a'..='g').contains(&note_char)
                                && key.kind == KeyEventKind::Press =>
                        {
                            let semitones =
                                [0, 2, -9, -7, -5, -4, -2][note_char as usize - 'a' as usize];
                            state.target_frequency =
                                Some(A4_FREQUENCY * compute_interval_ratio(semitones));
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
                                state.target_frequency = Some(freq * compute_interval_ratio(1));
                            }
                        }
                        event::KeyCode::Left if key.kind == KeyEventKind::Press => {
                            if let Some(freq) = state.target_frequency {
                                state.target_frequency = Some(freq * compute_interval_ratio(-1));
                            }
                        }
                        event::KeyCode::Esc if key.kind == KeyEventKind::Press => {
                            state.target_frequency = None;
                        }
                        event::KeyCode::Tab if key.kind == KeyEventKind::Press => {
                            state.graph = match state.graph {
                                Graph::AmplitudeSpectrum => Graph::CMNDF,
                                Graph::CMNDF => Graph::AmplitudeSpectrum,
                            }
                        }
                        _ => {}
                    },
                    Event::Mouse(e) if e.kind == MouseEventKind::Moved => {
                        state.cursor_pos = (e.row, e.column);
                    }
                    Event::Mouse(e) if e.kind == MouseEventKind::Drag(event::MouseButton::Left) => {
                        state.mouse_pressed = true;
                        state.cursor_pos = (e.row, e.column);
                    }
                    Event::Mouse(e) if e.kind == MouseEventKind::Up(event::MouseButton::Left) => {
                        state.mouse_pressed = false;
                    }
                    _ => {}
                }
            }
            _ => break,
        }
    }
}

fn initialize_panic_handler() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        crossterm::execute!(std::io::stderr(), crossterm::terminal::LeaveAlternateScreen).unwrap();
        crossterm::terminal::disable_raw_mode().unwrap();
        original_hook(panic_info);
    }));
}

fn main() -> io::Result<()> {
    initialize_panic_handler();
    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let host = cpal::default_host();
    let device = host.default_input_device().unwrap();
    let mut supported_configs_range = device
        .supported_input_configs()
        .expect("error while querying configs");
    let supported_config = supported_configs_range
        .next()
        .expect("no supported config?!")
        .with_max_sample_rate();

    let config = supported_config.config();
    let (sender, reciever) = std::sync::mpsc::channel::<f32>();
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // samples from different channels are received interleaved
                // mix all the channels into single one by averaging
                for b in data.chunks_exact(config.channels as usize) {
                    let val = b.iter().sum::<f32>() / f32::from(config.channels);
                    sender.send(val).unwrap();
                }
            },
            move |err| panic!("failed to get microphone input: {}", err),
            None,
        )
        .unwrap();
    stream.play().unwrap();

    let mut samples = Vec::new();
    let mut min_amount = BLOCK_SIZE.max(FFT_SIZE);

    let mut state = State::with_sample_rate(config.sample_rate.0 as f32);
    let mut avg_f0 = 0.0;
    let mut avg_db = 0.0;

    let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);

    while state.running {
        while let Ok(sample) = reciever.try_recv() {
            samples.push(sample);
        }

        if samples.len() >= min_amount {
            let mut fft_buf = samples[min_amount - FFT_SIZE..min_amount]
                .iter()
                .map(|&x| Complex { re: x, im: 0.0 })
                .collect::<Vec<_>>();

            if state.apply_window {
                apply_window(&mut fft_buf);
            }

            fft.process(&mut fft_buf);

            let cmndf = compute_cmndf(&samples[min_amount - BLOCK_SIZE..min_amount]);

            let f0 = compute_f0(&cmndf, state.sample_rate);
            avg_f0 = f32::exp(f32::ln(avg_f0) + 0.2 * (f32::ln(f0 / avg_f0)));
            if !avg_f0.is_finite() {
                avg_f0 = f0;
            }

            let rms = f32::sqrt(
                samples[min_amount - FFT_SIZE..min_amount]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    / BLOCK_SIZE as f32,
            );

            let db = 20.0 * f32::log10(rms);
            avg_db = avg_db + 0.4 * (db - avg_db);
            if !avg_db.is_finite() {
                avg_db = db;
            }

            let (note_index, cents) = if let Some(target_frequency) = state.target_frequency {
                let (target_note, _) = get_note(target_frequency);
                let cents = get_cents(avg_f0, target_frequency);
                (target_note, cents)
            } else {
                let (note, exact_frequency) = get_note(avg_f0);
                let cents = get_cents(avg_f0, exact_frequency);
                (note, cents)
            };

            terminal.draw(|f| {
                let main_layout = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(1)
                    .constraints(
                        [
                            Constraint::Min(7),
                            Constraint::Percentage(20),
                            Constraint::Percentage(80),
                        ]
                        .as_ref(),
                    )
                    .split(f.size());

                let info_layout = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(100), Constraint::Min(30)].as_ref())
                    .split(main_layout[0]);

                f.render_widget(
                    Paragraph::new(get_note_ascii(note_index).join("\n")),
                    info_layout[1],
                );

                draw_bar_chart(f, main_layout[1], avg_db, cents);

                draw_graph(f, main_layout[2], &state, &fft_buf, &cmndf, avg_f0);

                f.render_widget(
                    Paragraph::new(format!(
                        "Input device: {}\n\
                        Sample rate: {:.0}\n\
                        Target frequency: {}\n\
                        Detected frequency: {:.1}hz\n\
                        ",
                        device.name().unwrap(),
                        state.sample_rate,
                        match state.target_frequency {
                            Some(f) => format!("{:.1}hz", f),
                            None => "Not selected".into(),
                        },
                        avg_f0,
                    )),
                    info_layout[0],
                );

                let is_inside = state.cursor_pos.0 >= main_layout[2].top()
                    && state.cursor_pos.0 < main_layout[2].bottom()
                    && state.cursor_pos.1 >= main_layout[2].left()
                    && state.cursor_pos.1 < main_layout[2].right();

                if is_inside && state.mouse_pressed {
                    let low = match state.graph {
                        Graph::AmplitudeSpectrum => f32::ln(state.sample_rate / FFT_SIZE as f32),
                        Graph::CMNDF => f32::ln(state.sample_rate / (cmndf.len() - 1) as f32),
                    };
                    let high = f32::ln(state.sample_rate / 2.0);
                    let alpha = f32::from(state.cursor_pos.1 - main_layout[2].left())
                        / f32::from(main_layout[2].width - 1);
                    state.target_frequency = Some(f32::exp(lerp(low, high, alpha)));
                }
            })?;

            handle_events(&mut state);

            min_amount += BLOCK_SIZE / 2;
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
