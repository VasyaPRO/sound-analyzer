use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    prelude::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Bar, BarChart, Block, BorderType, Borders, Paragraph, BarGroup},
    Terminal,
};
use std::{
    io, time::Duration,
};
// use std::io;
// use ratatui::{backend::CrosstermBackend, Terminal};

const WINDOW_SIZE: usize = 1024;

// auto correlaction function
// fn acf(x: &[f32], lag: usize) -> f32 {
//     let mut res = 0.0;
//     for i in 0..WINDOW_SIZE {
//         res += x[i] * x[i + lag];
//     }
//     return res;
// }

fn parabolic_interpolation(u: f32, v: f32, w: f32) -> f32 {
    0.5 * (u - w) / (u - 2.0 * v + w)
}

// float sumDifferenceSquaredValues[WINDOW_SIZE];
// float CMNDF_values[]
// cumulative mean normalized difference function
// float CMNDFValues[WINDOW_SIZE];

fn compute_cmndf(x: &[f32]) -> Vec<f32> {
    // dbg!("cmndf len", x.len());
    let samples_count = x.len() / 2;
    // dbg!(x.len(), samples_count);
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

fn get_note(frequency: f32) -> (String, f32) {
    const A4_FREQUENCY: f32 = 440.0;
    let semitones = f32::log2(frequency / A4_FREQUENCY) * 12.0;
    const NOTES: [&str; 12] = [
        "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
    ];
    let note = semitones.round() as i32;
    let octave = note.div_euclid(12) + 4;
    let exact_frequency = A4_FREQUENCY * 2.0_f32.powf(note as f32 / 12.0);
    let cents = f32::log2(frequency / exact_frequency) * 12.0 * 100.0;
    // freq / exact = 2^(0.5 / 12)
    // log2(freq / exact) = 0.5 / 12
    // 0.5 = 12 * log(freq / exact)
    (
        NOTES[note.rem_euclid(12) as usize].to_owned() + &octave.to_string(),
        cents,
    )
}

// fn main() -> Result<(), io::Error> {
//     // setup terminal
//     enable_raw_mode()?;
//     let mut stdout = io::stdout();
//     execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
//     let backend = CrosstermBackend::new(stdout);
//     let mut terminal = Terminal::new(backend)?;
//
//     let mut perc = 0;
//
//     let mut running = true;
//     // Start a thread to discard any input events. Without handling events, the
//     // stdin buffer will fill up, and be read into the shell when the program exits.
//     // thread::spawn(|| loop {
//     //     dbg!(ev);
//     // });
//     //
//     //
//     thread::spawn(move || {
//         let mut last_tick = Instant::now();
//         loop {
//             // let timeout = tick_rate
//             //     .checked_sub(last_tick.elapsed())
//             //     .unwrap_or(tick_rate);
//
//             match event::read().unwrap() {
//                 // match event::read().expect("unable to read event") {
//                     Event::Key(e) =>  println!("key {:?}", e),
//                     Event::Mouse(e) => println!("mouse {:?}", e),
//                     Event::Resize(w, h) => println!("resize {}x{}", w, h),
//                     x => { println!("resize {:?}", x); unimplemented!()}
//             }
//             // if event::poll(timeout).expect("no events available") {
//             //     match event::read().expect("unable to read event") {
//             //         CrosstermEvent::Key(e) => sender.send(Event::Key(e)),
//             //         CrosstermEvent::Mouse(e) => sender.send(Event::Mouse(e)),
//             //         CrosstermEvent::Resize(w, h) => sender.send(Event::Resize(w, h)),
//             //         _ => unimplemented!(),
//             //     }
//             //         .expect("failed to send terminal event")
//             // }
//             //
//             // if last_tick.elapsed() >= tick_rate {
//             //     sender.send(Event::Tick).expect("failed to send tick event");
//             //     last_tick = Instant::now();
//             // }
//         }
//     });
//     // let ev = tui.event::next()?;
//     while running {
//
//         terminal.draw(|f| {
//             // let size = f.size();
//             // let block = Block::default()
//             //     .title("Block")
//             //     .borders(Borders::ALL);
//             // let span = ratatui::text::Span::from("hello world");
//             // let chunks = Layout::default()
//             //     .direction(Direction::Vertical)
//             //     .margin(2)
//             //     .constraints([Constraint::Percentage(100)].as_ref())
//             //     .split(f.size());
//             //
//             // let hello_text = Text::styled("Hello, World!", Style::default().fg(Color::Yellow));
//             //
//             // let block = Block::default().title("Hello World").borders(Borders::ALL);
//             //
//             // let paragraph = Paragraph::new(hello_text)
//             //     .block(block)
//             //     .alignment(Alignment::Center);
//             //
//             // let gauge = Gauge::default()
//             //     .block(Block::default().borders(Borders::ALL).title("Progress"))
//             //     .gauge_style(Style::default().fg(Color::White).bg(Color::Black).add_modifier(Modifier::ITALIC))
//             //     .percent(perc);
//             // perc += 1;
//             // perc %= 100;
//             //
//             //
//             // f.render_widget(paragraph, chunks[0]);
//             // f.render_widget(gauge, chunks[0]);
//             // f.render_widget(block, size);
//             // f.render_widget(span, size);
//             //
//
//             f.render_widget(
//                 Paragraph::new(format!(
//                 "This is a tui template.\n\
//                 Press `Esc`, `Ctrl-C` or `q` to stop running.\n\
//                 Press left and right to increment and decrement the counter respectively.\n\
//                 Counter: {}",
//                 42
//             ))
//                     .block(
//                         Block::default()
//                             .title("Template")
//                             .title_alignment(Alignment::Center)
//                             .borders(Borders::ALL)
//                             .border_type(BorderType::Rounded),
//                     )
//                     .style(Style::default().fg(Color::Cyan).bg(Color::Black))
//                     .alignment(Alignment::Center),
//                 f.size(),
//             )
//         })?;
//
//
//
//         // if let Event::Key(k) = ev? {
//         //
//         // }
//         //
//
//         // match key_event.code {
//         //     // Exit application on `ESC` or `q`
//         //     KeyCode::Esc | KeyCode::Char('q') => {
//         //         app.quit();
//         //     }
//         //     // Exit application on `Ctrl-C`
//         //     KeyCode::Char('c') | KeyCode::Char('C') => {
//         //         if key_event.modifiers == KeyModifiers::CONTROL {
//         //             app.quit();
//         //         }
//         //     }
//         //     // Counter handlers
//         //     KeyCode::Right => {
//         //         app.increment_counter();
//         //     }
//         //     KeyCode::Left => {
//         //         app.decrement_counter();
//         //     }
//         //     // Other handlers you could add here.
//         //     _ => {}
//         // }
//         thread::sleep(Duration::from_millis(16));
//     }
//
//
//     // restore terminal
//     disable_raw_mode()?;
//     execute!(
//         terminal.backend_mut(),
//         LeaveAlternateScreen,
//         DisableMouseCapture
//     )?;
//     terminal.show_cursor()?;
//
//     Ok(())
// }

fn main() -> io::Result<()> {
    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let host = cpal::default_host();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(c) = device.supported_input_configs() {
                for x in c {
                    dbg!(x);
                }
            }
        }
    }
    let device = host.default_input_device().unwrap();
    let mut supported_configs_range = device
        .supported_input_configs()
        .expect("error while querying configs");
    println!("Input device: {:?}", device.name());
    let supported_config = supported_configs_range
        .next()
        .expect("no supported config?!")
        // .with_sample_rate(cpal::SampleRate(48000));
        .with_max_sample_rate();
    let config = supported_config.config();
    println!("hello");
    // dbg!(device);
    // let cnt = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // let cnt2 = cnt.clone();
    let (sender, reciever) = std::sync::mpsc::channel::<f32>();
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // sample from two channels are received interleaved
                // we only take one channel now
                for i in (0..data.len()).step_by(2) {
                    sender.send(data[i]).unwrap();
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
    let mut min_amount = BLOCK_SIZE;
    let mut frequencies = Vec::new();

    let mut running = true;
    let mut avg_f0 = 0.0;
    let mut avg_db = 0.0;
    while running {
        while let Ok(sample) = reciever.try_recv() {
            assert!(f32::abs(sample) <= 1.0);
            samples.push(sample);
        }

        if samples.len() >= min_amount {
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
            // avg_f0 = avg_f0 + 0.1 * (f0 - avg_f0);
            avg_f0 = f0;
            // println!("{}hz", f0);
            // println!("{}rms mn_pos{}", rms, min_pos);
            // println!("{}db", 20.0 * f32::log10(rms));
            let (note, cents) = get_note(avg_f0);
            // println!("{} {:+}", note, cents);
            frequencies.push(f0);
            min_amount += BLOCK_SIZE / 4;

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
                            Constraint::Percentage(20),
                            Constraint::Percentage(80),
                            // Constraint::Percentage(10)
                        ]
                        .as_ref(),
                    )
                    .split(f.size());
                let db = 20.0 * f32::log10(rms);

                // avg_db = avg_db + 0.1 * (db - avg_db);
                avg_db = db;
                f.render_widget(
                    Paragraph::new(format!(
                        "Frequency: {:9.3}\n\
                    Note: {:3} {:+2.3}\n\
                    {:.3} db\n\
                    Press `Esc`, `Ctrl-C` or `q` to stop running.",
                        avg_f0, note, cents, db
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
                    layout[0],
                );


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
            })?;

            loop {
                match event::poll(Duration::from_millis(0)) {
                    Ok(ready) if ready => {
                        let ev = event::read().unwrap();
                        match ev {
                            Event::Key(key) if key.modifiers == KeyModifiers::CONTROL => {
                                //println!("key {:?}", key);
                                match key.code {
                                    event::KeyCode::Char('c') | event::KeyCode::Char('C') => {
                                        running = false
                                    }
                                    _ => {}
                                }
                            }
                            // Event::Mouse(e) => println!("mouse {:?}", e),
                            // Event::Resize(w, h) => println!("resize {}x{}", w, h),
                            _ => {}
                        }
                    }
                    _ => break,
                }
            }
        }

        // thread::sleep(Duration::from_millis(16));
    }

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    // reciever.recv();
    // std::thread::sleep(std::time::Duration::new(1, 0));
    // dbg!(cnt.load(std::sync::atomic::Ordering::SeqCst));
    // cnt.store(0, std::sync::atomic::Ordering::SeqCst);
    // }
    // dbg!(samples.len());
    // println!("Hello, world!");

    // if false {
    //     let device = host.default_output_device().unwrap();
    //     let mut supported_configs_range = device
    //         .supported_output_configs()
    //         .expect("error while querying configs");
    //     println!("Output device: {:?}", device.name());
    //     let supported_config = supported_configs_range
    //         .next()
    //         .expect("no supported config?!")
    //         .with_max_sample_rate();
    //     let config = supported_config.config();
    //     let stream = device
    //         .build_output_stream(
    //             &config,
    //             move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
    //                 // while let Ok(sample) = reciever.try_recv() {
    //                 //     samples.push_back(sample);
    //                 // }
    //                 for x in data.iter_mut() {
    //                     match reciever.try_recv() {
    //                         Ok(sample) => *x = sample,
    //                         Err(_) => *x = 0.0,
    //                     }
    //                     // *x =
    //                 }
    //                 // cnt2.fetch_add(data.len().into(), std::sync::atomic::Ordering::SeqCst);
    //                 // for sample in data[0..data.len() / 2].iter() {
    //                 //     sender.send(*sample).unwrap();
    //                 // }
    //                 // dbg!(data.len());
    //                 // println!("hello");
    //             },
    //             move |err| {
    //                 dbg!(err);
    //                 println!("err output");
    //             },
    //             None, // None=blocking, Some(Duration)=timeout
    //         )
    //         .unwrap();
    //     stream.play().unwrap();
    //     std::thread::sleep(std::time::Duration::new(3, 0));
    //     drop(stream);
    // }
    Ok(())
}

// fn main() {
//     let host = cpal::default_host();
//     if let Ok(input_devices) = host.input_devices() {
//         for device in input_devices {
//             if let Ok(c) = device.supported_input_configs() {
//                 for x in c {
//                     dbg!(x);
//                 }
//             }
//         }
//     }
//     let device = host.default_input_device().unwrap();
//     let mut supported_configs_range = device
//         .supported_input_configs()
//         .expect("error while querying configs");
//     println!("Input device: {:?}", device.name());
//     let supported_config = supported_configs_range
//         .next()
//         .expect("no supported config?!")
//         // .with_sample_rate(cpal::SampleRate(48000));
//         .with_max_sample_rate();
//     let config = supported_config.config();
//     println!("hello");
//     // dbg!(device);
//     // let cnt = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
//     // let cnt2 = cnt.clone();
//     let (sender, reciever) = std::sync::mpsc::channel::<f32>();
//     let stream = device
//         .build_input_stream(
//             &config,
//             move |data: &[f32], _: &cpal::InputCallbackInfo| {
//                 // cnt2.fetch_add(data.len().into(), std::sync::atomic::Ordering::SeqCst);
//                 // for sample in data[0..data.len() / 2].iter() {
//                 // for sample in data[0..data.len() / 2].iter() {
//                 //
//                 // sample from two channels are received interleaved
//                 // we only take one channel now
//                 for i in (0..data.len()).step_by(2) {
//                     sender.send(data[i]).unwrap();
//                 }
//                 // for sample in data.iter() {
//                 //     sender.send(*sample).unwrap();
//                 // }
//                 // dbg!(data.len());
//                 // println!("hello");
//             },
//             move |err| {
//                 dbg!(err);
//                 println!("err");
//             },
//             None, // None=blocking, Some(Duration)=timeout
//         )
//         .unwrap();
//     stream.play().unwrap();
//     // stream.
//     // std::thread::sleep(std::time::Duration::new(3, 0));
//     // drop(stream);
//     let mut samples = Vec::new();
//     const BLOCK_SIZE: usize = WINDOW_SIZE * 2;
//     let mut min_amount = BLOCK_SIZE;
//     let mut frequencies = Vec::new();
//     loop {
//         while let Ok(sample) = reciever.try_recv() {
//             assert!(f32::abs(sample) <= 1.0);
//             samples.push(sample);
//         }
//         if samples.len() >= min_amount {
//             // let freq: f32 = 0.0;
//             dbg!(min_amount - BLOCK_SIZE, min_amount, samples.len());
//             let cmndf = compute_cmndf(&samples[min_amount - BLOCK_SIZE..min_amount]);
//             let rms = f32::sqrt(samples[min_amount - BLOCK_SIZE..min_amount].iter().map(|x| x * x).sum::<f32>() / BLOCK_SIZE as f32);
//             // frequencies.push(freq);
//             let mut min_pos = 1;
//             while min_pos < WINDOW_SIZE - 1 {
//                 if cmndf[min_pos] > 0.03 {
//                     min_pos += 1;
//                 } else {
//                     break;
//                 }
//             }
//             if min_pos == WINDOW_SIZE - 1 {
//                 min_pos = cmndf[1..WINDOW_SIZE - 1]
//                     .iter()
//                     .enumerate()
//                     .min_by(|(_, a), (_, b)| a.total_cmp(b))
//                     .map(|(index, _)| index)
//                     .unwrap()
//                     + 1;
//             }
//             // let sample_rate = 48000;
//             // let sr = config.sample_rate;
//             dbg!(config.sample_rate);
//             let f0: f32 = config.sample_rate.0 as f32
//                 / (min_pos as f32
//                     + parabolic_interpolation(
//                         cmndf[min_pos - 1],
//                         cmndf[min_pos],
//                         cmndf[min_pos + 1],
//                     ));
//             println!("{}hz", f0);
//             println!("{}rms mn_pos{}", rms, min_pos);
//             println!("{}db", 20.0 * f32::log10(rms));
//             let (note, cents) = get_note(f0);
//             println!("{} {:+}", note, cents);
//             frequencies.push(f0);
//             min_amount += BLOCK_SIZE;
//         }
//     }
//     // reciever.recv();
//     // std::thread::sleep(std::time::Duration::new(1, 0));
//     // dbg!(cnt.load(std::sync::atomic::Ordering::SeqCst));
//     // cnt.store(0, std::sync::atomic::Ordering::SeqCst);
//     // }
//     // dbg!(samples.len());
//     // println!("Hello, world!");
//
//     // if false {
//     //     let device = host.default_output_device().unwrap();
//     //     let mut supported_configs_range = device
//     //         .supported_output_configs()
//     //         .expect("error while querying configs");
//     //     println!("Output device: {:?}", device.name());
//     //     let supported_config = supported_configs_range
//     //         .next()
//     //         .expect("no supported config?!")
//     //         .with_max_sample_rate();
//     //     let config = supported_config.config();
//     //     let stream = device
//     //         .build_output_stream(
//     //             &config,
//     //             move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
//     //                 // while let Ok(sample) = reciever.try_recv() {
//     //                 //     samples.push_back(sample);
//     //                 // }
//     //                 for x in data.iter_mut() {
//     //                     match reciever.try_recv() {
//     //                         Ok(sample) => *x = sample,
//     //                         Err(_) => *x = 0.0,
//     //                     }
//     //                     // *x =
//     //                 }
//     //                 // cnt2.fetch_add(data.len().into(), std::sync::atomic::Ordering::SeqCst);
//     //                 // for sample in data[0..data.len() / 2].iter() {
//     //                 //     sender.send(*sample).unwrap();
//     //                 // }
//     //                 // dbg!(data.len());
//     //                 // println!("hello");
//     //             },
//     //             move |err| {
//     //                 dbg!(err);
//     //                 println!("err output");
//     //             },
//     //             None, // None=blocking, Some(Duration)=timeout
//     //         )
//     //         .unwrap();
//     //     stream.play().unwrap();
//     //     std::thread::sleep(std::time::Duration::new(3, 0));
//     //     drop(stream);
//     // }
// }
