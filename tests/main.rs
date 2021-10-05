use esolangs::brainfuck::prelude::*;

#[test]
fn mandelbrot() {
    brainfuck_include!("tests/bf/mandelbrot.b");
}

#[test]
fn numwarp() {
    brainfuck_include!("tests/bf/numwarp.b");
}

// http://brainfuck.org/tests.b

#[test]
fn test_1() {
    brainfuck! {
        >,>+++++++++,>+++++++++++[<++++++<++++++<+>>>-]<<.>.<<-.>.>.<<.
    }
}

#[test]
fn test_2() {
    brainfuck! {
        ++++[>++++++<-]>[>+++++>+++++++<<-]>>++++<[[>[[>>+<<-]<]>>>-]>-[>+>+<<-]>]
        +++++[>+++++++<<++>-]>.<<.
    }
}

#[test]
fn test_3() {
    brainfuck! {
        []++++++++++[>>+>+>++++++[<<+<+++>>>-]<<<<-]
        "A*$";?@![#>>+<<]>[>>]<<<<[>++<[-]]>.>.
    }
}
