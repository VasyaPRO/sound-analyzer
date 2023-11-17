const ASCII_LETTERS: [[&str; 7]; 17] = [
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


pub fn get_note_ascii(note: isize) -> Vec<String> {
    let octave = note.div_euclid(12) +
        if note.rem_euclid(12) <= 2 {
            // A to B
            4
        } else {
            // C to G#
            5
        };
    let octave = octave.clamp(1, 9) as usize;

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

