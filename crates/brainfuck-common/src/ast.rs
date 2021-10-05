use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct Ast {
    instructions: Vec<Instruction>,
}

impl Ast {
    pub(crate) fn optimize(&mut self) {
        self.instructions = optimize_instructions(&self.instructions);
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            instructions: parse_bytes(bytes),
        }
    }
}

fn parse_bytes(bytes: &[u8]) -> Vec<Instruction> {
    let mut instructions: Vec<Instruction> = Vec::with_capacity(bytes.len());
    let mut i: usize = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'>' => {
                instructions.push(Instruction::MovePtr(MovePtr {
                    offset: 1,
                    forward: true,
                }));
                i += 1;
            }
            b'<' => {
                instructions.push(Instruction::MovePtr(MovePtr {
                    offset: 1,
                    forward: false,
                }));
                i += 1;
            }
            b'+' => {
                instructions.push(Instruction::IncrementCell(IncrementCell { amount: 1 }));
                i += 1;
            }
            b'-' => {
                instructions.push(Instruction::IncrementCell(IncrementCell { amount: 255 }));
                i += 1;
            }
            b'.' => {
                instructions.push(Instruction::PrintCell(PrintCell {}));
                i += 1;
            }
            b',' => {
                instructions.push(Instruction::ReadToCell(ReadToCell {}));
                i += 1;
            }
            b'#' => {
                instructions.push(Instruction::Debug(Debug {}));
                i += 1;
            }
            b'[' => {
                let mut nesting: usize = 1;
                let mut inner_bytes = vec![];
                i += 1;
                while nesting > 0 {
                    inner_bytes.push(bytes[i]);
                    match bytes[i] {
                        b'[' => nesting += 1,
                        b']' => nesting -= 1,
                        _ => (),
                    }
                    i += 1;
                }
                instructions.push(Instruction::Loop(Loop {
                    inner: parse_bytes(&inner_bytes),
                }))
            }
            _ => i += 1,
        }
    }
    instructions
}
fn optimize_instructions(input: &[Instruction]) -> Vec<Instruction> {
    println!("Optimize");
    println!(" Input: {:?}", input);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let mut output: Vec<Instruction> = Vec::with_capacity(input.len());
    for instr in input {
        if let Some(prev_instr) = output.last_mut() {
            match (prev_instr, instr) {
                (_, Instruction::IncrementCell(IncrementCell { amount: 0 })) => {}
                (_, Instruction::MovePtr(MovePtr { offset: 0, .. })) => {}
                (Instruction::MovePtr(prev_move), Instruction::MovePtr(curr_move)) => {
                    if prev_move.forward == curr_move.forward {
                        if let Some(sum) = prev_move.offset.checked_add(curr_move.offset) {
                            prev_move.offset = sum;
                        } else {
                            output.push(instr.clone());
                        }
                    } else {
                        match prev_move.offset.cmp(&curr_move.offset) {
                            std::cmp::Ordering::Less => {
                                prev_move.forward = !prev_move.forward;
                                prev_move.offset = curr_move.offset - prev_move.offset;
                            }
                            std::cmp::Ordering::Equal => {
                                output.pop();
                            }
                            std::cmp::Ordering::Greater => {
                                prev_move.offset -= curr_move.offset;
                            }
                        }
                    }
                }
                (Instruction::IncrementCell(prev_incr), Instruction::IncrementCell(incr)) => {
                    prev_incr.amount = prev_incr.amount.wrapping_add(incr.amount);
                    if prev_incr.amount == 0 {
                        println!("whee");
                        output.pop();
                    }
                }
                (Instruction::IncrementCell(_), Instruction::SetCell(_)) => {
                    output.pop();
                    output.push(instr.clone());
                }
                (Instruction::SetCell(prev_set), Instruction::IncrementCell(incr)) => {
                    prev_set.value = prev_set.value.wrapping_add(incr.amount);
                }
                (Instruction::SetCell(_), Instruction::SetCell(_)) => {
                    output.pop();
                    output.push(instr.clone());
                }
                (Instruction::SetCell(prev_set), Instruction::PrintCell(_)) => {
                    let value = prev_set.value;
                    output.push(Instruction::PrintValue(PrintValue { value }));
                }
                (Instruction::SetCell(SetCell { value: 0 }), Instruction::Loop(_)) => {}
                (Instruction::Loop(_), Instruction::IncrementCell(incr)) => {
                    output.push(Instruction::SetCell(SetCell { value: incr.amount }));
                }
                (Instruction::Loop(_), Instruction::SetCell(set)) => {
                    if set.value != 0 {
                        output.push(instr.clone());
                    }
                }
                (Instruction::Loop(_), Instruction::PrintCell(_)) => {
                    output.push(Instruction::PrintValue(PrintValue { value: 0 }));
                }
                (Instruction::Loop(_), Instruction::Loop(_)) => {}
                (_, Instruction::Loop(curr_loop)) => {
                    let optimized_inner = optimize_instructions(&curr_loop.inner);
                    if optimized_inner.len() == 1 {
                        match optimized_inner[0] {
                            Instruction::IncrementCell(incr) => {
                                if incr.amount == 0 {
                                    output.push(Instruction::Loop(Loop {
                                        inner: Vec::with_capacity(0),
                                    }));
                                } else {
                                    output.push(Instruction::SetCell(SetCell { value: 0 }));
                                }
                            }
                            Instruction::SetCell(set_cell) => {
                                if set_cell.value == 0 {
                                    output.push(Instruction::SetCell(SetCell { value: 0 }));
                                } else {
                                    output.push(Instruction::Loop(Loop {
                                        inner: optimized_inner,
                                    }));
                                }
                            }
                            _ => {
                                output.push(Instruction::Loop(Loop {
                                    inner: optimized_inner,
                                }));
                            }
                        }
                    } else {
                        output.push(Instruction::Loop(Loop {
                            inner: optimized_inner,
                        }));
                    }
                }
                _ => output.push(instr.clone()),
            }
        } else {
            output.push(instr.clone());
        }
    }

    println!("Output: {:?}", output);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    if input == output {
        output
    } else {
        optimize_instructions(&output)
    }
}

impl ToTokens for Ast {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut instr_tokens = TokenStream::new();
        for instr in &self.instructions {
            instr.to_tokens(&mut instr_tokens);
        }

        (quote! {
            let _ = {
                let mut memory = __bf_macro_tiny_vec!([u8; 30000]);
                memory.resize(1, 0);
                let mut ptr: usize = 0;
                let mut input = ::std::io::stdin();
                let mut output = ::std::io::stdout();

                #instr_tokens
            };
        })
        .to_tokens(tokens);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Instruction {
    MovePtr(MovePtr),
    IncrementCell(IncrementCell),
    SetCell(SetCell),
    ReadToCell(ReadToCell),
    PrintCell(PrintCell),
    PrintValue(PrintValue),
    Loop(Loop),
    Debug(Debug),
}

impl ToTokens for Instruction {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Instruction::MovePtr(instr) => instr.to_tokens(tokens),
            Instruction::IncrementCell(instr) => instr.to_tokens(tokens),
            Instruction::SetCell(instr) => instr.to_tokens(tokens),
            Instruction::ReadToCell(instr) => instr.to_tokens(tokens),
            Instruction::PrintCell(instr) => instr.to_tokens(tokens),
            Instruction::PrintValue(instr) => instr.to_tokens(tokens),
            Instruction::Loop(instr) => instr.to_tokens(tokens),
            Instruction::Debug(instr) => instr.to_tokens(tokens),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct MovePtr {
    offset: usize,
    forward: bool,
}

impl ToTokens for MovePtr {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let offset = self.offset;
        if offset != 0 {
            if self.forward {
                (quote! {
                    ptr = ptr.checked_add(#offset).unwrap();
                    memory.resize(::core::cmp::max(memory.len(), ptr.wrapping_add(1)), 0);
                })
                .to_tokens(tokens);
            } else {
                (quote! {
                    ptr = ptr.checked_sub(#offset).unwrap();
                    memory.resize(::core::cmp::max(memory.len(), ptr.wrapping_add(1)), 0);
                })
                .to_tokens(tokens);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct IncrementCell {
    amount: u8,
}

impl ToTokens for IncrementCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let amount = self.amount;
        if amount != 0 {
            (quote! {
                memory[ptr] = memory[ptr].wrapping_add(#amount);
            })
            .to_tokens(tokens);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SetCell {
    value: u8,
}

impl ToTokens for SetCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let value = self.value;
        (quote! {
            memory[ptr] = #value;
        })
        .to_tokens(tokens);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ReadToCell {}

impl ToTokens for ReadToCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        (quote! {
            let mut buf = [0u8];
            if ::std::io::Read::read_exact(&mut input, &mut buf).is_ok() {
                memory[ptr] = buf[0];
            }
        })
        .to_tokens(tokens);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct PrintCell {}

impl ToTokens for PrintCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        (quote! {
            if let Err(err) = ::std::io::Write::write(&mut output, &[memory[ptr]]) {
                if err.kind() == ::std::io::ErrorKind::Interrupted {
                    let _ = ::std::io::Write::write(&mut output, &[memory[ptr]]);
                }
            }
            let _ = ::std::io::Write::flush(&mut output);
        })
        .to_tokens(tokens);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct PrintValue {
    value: u8,
}

impl ToTokens for PrintValue {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let value = self.value;
        (quote! {
            if let Err(err) = ::std::io::Write::write(&mut output, &[#value]) {
                if err.kind() == ::std::io::ErrorKind::Interrupted {
                    let _ = ::std::io::Write::write(&mut output, &[#value]);
                }
            }
            let _ = ::std::io::Write::flush(&mut output);
        })
        .to_tokens(tokens);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Loop {
    inner: Vec<Instruction>,
}

impl ToTokens for Loop {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let mut inner_tokens = TokenStream::new();

        for instr in &self.inner {
            instr.to_tokens(&mut inner_tokens);
        }

        (quote! {
            while memory[ptr] != 0 {
                #inner_tokens
            }
        })
        .to_tokens(tokens);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Debug {}

impl ToTokens for Debug {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        (quote! {
            println!("Pointer: {} | Memory: {}", ptr, memory);
        })
        .to_tokens(tokens);
    }
}
