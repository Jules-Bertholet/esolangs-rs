use std::{
    cmp,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

// Copied from core lib /src/internal_macros.rs
macro_rules! forward_ref_unop {
    (impl $imp:ident, $method:ident for $t:ty) => {
        impl $imp for &$t {
            type Output = <$t as $imp>::Output;

            #[inline]
            fn $method(self) -> <$t as $imp>::Output {
                $imp::$method(*self)
            }
        }
    };
}

macro_rules! forward_ref_binop {
    (impl $imp:ident, $method:ident for $t:ty, $u:ty) => {
        impl<'a> $imp<$u> for &'a $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(*self, other)
            }
        }

        impl $imp<&$u> for $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: &$u) -> <$t as $imp<$u>>::Output {
                $imp::$method(self, *other)
            }
        }

        impl $imp<&$u> for &$t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: &$u) -> <$t as $imp<$u>>::Output {
                $imp::$method(*self, *other)
            }
        }
    };
}

macro_rules! forward_ref_op_assign {
    (impl $imp:ident, $method:ident for $t:ty, $u:ty) => {
        impl $imp<&$u> for $t {
            #[inline]
            fn $method(&mut self, other: &$u) {
                $imp::$method(self, *other);
            }
        }
    };
}

/// This type represents a pointer offset, for shifts or offset instructions.
/// It is basically a `usize` with an extra sign bit.
#[derive(Clone, Copy, Debug)]
enum PtrOffset {
    Backward(usize),
    Zero,
    Forward(usize),
}

impl Default for PtrOffset {
    fn default() -> Self {
        PtrOffset::Zero
    }
}

impl From<usize> for PtrOffset {
    fn from(val: usize) -> Self {
        if val == 0 {
            Self::Zero
        } else {
            Self::Forward(val)
        }
    }
}

impl Neg for PtrOffset {
    type Output = PtrOffset;

    fn neg(self) -> Self::Output {
        match self {
            Self::Backward(val) => Self::Forward(val),
            Self::Zero => Self::Zero,
            Self::Forward(val) => Self::Backward(val),
        }
    }
}

forward_ref_unop!(impl Neg, neg for PtrOffset);

impl Add for PtrOffset {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Forward(l0), Self::Forward(r0)) => Self::Forward(l0.checked_add(r0).unwrap()),
            (Self::Backward(l0), Self::Backward(r0)) => Self::Backward(l0.checked_add(r0).unwrap()),
            (Self::Forward(l0), Self::Backward(r0)) => match l0.cmp(&r0) {
                cmp::Ordering::Less => Self::Backward(r0 - l0),
                cmp::Ordering::Equal => Self::Zero,
                cmp::Ordering::Greater => Self::Forward(l0 - r0),
            },
            (Self::Backward(l0), Self::Forward(r0)) => match l0.cmp(&r0) {
                cmp::Ordering::Less => Self::Forward(r0 - l0),
                cmp::Ordering::Equal => Self::Zero,
                cmp::Ordering::Greater => Self::Backward(l0 - r0),
            },
            (Self::Backward(_) | Self::Forward(_), Self::Zero) => self,
            (Self::Zero, Self::Backward(_) | Self::Forward(_)) => rhs,
            (Self::Zero, Self::Zero) => Self::Zero,
        }
    }
}

forward_ref_binop!(impl Add, add for PtrOffset, PtrOffset);

impl Add<usize> for PtrOffset {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        self + Self::from(rhs)
    }
}

forward_ref_binop!(impl Add, add for PtrOffset, usize);

impl Sub for PtrOffset {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

forward_ref_binop!(impl Sub, sub for PtrOffset, PtrOffset);

impl Sub<usize> for PtrOffset {
    type Output = PtrOffset;

    fn sub(self, rhs: usize) -> Self::Output {
        self - Self::from(rhs)
    }
}

forward_ref_binop!(impl Sub, sub for PtrOffset, usize);

impl AddAssign for PtrOffset {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

forward_ref_op_assign!(impl AddAssign, add_assign for PtrOffset, PtrOffset);

impl AddAssign<usize> for PtrOffset {
    fn add_assign(&mut self, rhs: usize) {
        *self = *self + Self::from(rhs);
    }
}

forward_ref_op_assign!(impl AddAssign, add_assign for PtrOffset, usize);

impl SubAssign for PtrOffset {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

forward_ref_op_assign!(impl SubAssign, sub_assign for PtrOffset, PtrOffset);

impl SubAssign<usize> for PtrOffset {
    fn sub_assign(&mut self, rhs: usize) {
        *self = *self - Self::from(rhs);
    }
}

forward_ref_op_assign!(impl SubAssign, sub_assign for PtrOffset, usize);

impl PartialEq for PtrOffset {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Zero, Self::Zero) => true,
            (Self::Backward(l0), Self::Backward(r0)) | (Self::Forward(l0), Self::Forward(r0)) => {
                l0 == r0
            }
            _ => false,
        }
    }
}

impl Eq for PtrOffset {}

impl PartialOrd for PtrOffset {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PtrOffset {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match (self, other) {
            (PtrOffset::Backward(_), PtrOffset::Zero | PtrOffset::Forward(_))
            | (PtrOffset::Zero, PtrOffset::Forward(_)) => cmp::Ordering::Less,
            (PtrOffset::Backward(l0), PtrOffset::Backward(r0)) => r0.cmp(l0),
            (PtrOffset::Zero, PtrOffset::Zero) => cmp::Ordering::Equal,
            (PtrOffset::Forward(l0), PtrOffset::Forward(r0)) => l0.cmp(r0),
            (PtrOffset::Forward(_), PtrOffset::Zero | PtrOffset::Backward(_))
            | (PtrOffset::Zero, PtrOffset::Backward(_)) => cmp::Ordering::Greater,
        }
    }
}

impl ToTokens for PtrOffset {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (match self {
            PtrOffset::Backward(val) => quote! {.checked_sub(#val).unwrap() },
            PtrOffset::Zero => quote! {},
            PtrOffset::Forward(val) => quote! { .checked_add(#val).unwrap() },
        })
        .to_tokens(tokens);
    }
}

/// This enum represents all the possible instruction types.
/// The AST we use is a Vec<Instruction>
#[derive(Clone, PartialEq, Eq, Debug)]
enum Instruction {
    ProgramStart(ProgramStart),
    ShiftPtr(ShiftPtr),
    ScanLoop(ScanLoop),
    IncrementCell(IncrementCell),
    SetCell(SetCell),
    MultiplyToCell(MultiplyToCell),
    ReadToCell(ReadToCell),
    PrintCell(PrintCell),
    PrintValue(PrintValue),
    Loop(Loop),
    Debug(Debug),
}

impl ToTokens for Instruction {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Instruction::ProgramStart(instr) => instr.to_tokens(tokens),
            Instruction::ShiftPtr(instr) => instr.to_tokens(tokens),
            Instruction::ScanLoop(scan) => scan.to_tokens(tokens),
            Instruction::IncrementCell(instr) => instr.to_tokens(tokens),
            Instruction::SetCell(instr) => instr.to_tokens(tokens),
            Instruction::MultiplyToCell(instr) => instr.to_tokens(tokens),
            Instruction::ReadToCell(instr) => instr.to_tokens(tokens),
            Instruction::PrintCell(instr) => instr.to_tokens(tokens),
            Instruction::PrintValue(instr) => instr.to_tokens(tokens),
            Instruction::Loop(instr) => instr.to_tokens(tokens),
            Instruction::Debug(instr) => instr.to_tokens(tokens),
        }
    }
}

/// A special instruction for the start of the program.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct ProgramStart {}

impl ToTokens for ProgramStart {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (quote! {
            let mut memory = ::esolangs::brainfuck::macro_reexports::tinyvec::tiny_vec!([u8; 30000]);
            memory.resize(1, 0);
            let mut ptr: usize = 0;
            let mut input = ::std::io::stdin();
            let mut output = ::std::io::stdout();
            let mut buf = [0u8];
        })
        .to_tokens(tokens);
    }
}

/// `<`, `>`, and combinations thereof.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ShiftPtr {
    offset: PtrOffset,
}

impl ToTokens for ShiftPtr {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let offset = self.offset;
        match offset {
            PtrOffset::Backward(_) => {
                (quote! {
                    ptr = ptr #offset;
                })
                .to_tokens(tokens);
            }
            PtrOffset::Forward(_) => {
                (quote! {
                    ptr = ptr #offset;
                    memory.resize(::core::cmp::max(memory.len(), ptr.checked_add(1).unwrap()), 0);
                })
                .to_tokens(tokens);
            }
            PtrOffset::Zero => {}
        }
    }
}

/// `<`, `>`, and combinations thereof.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ScanLoop {
    value: u8,
    forward: bool,
}

impl ToTokens for ScanLoop {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let value = self.value;
        if self.forward {
            (quote! {
                if let Some(offset) = ::esolangs::brainfuck::macro_reexports::memchr::memchr(#value, &(&memory)[ptr..]) {
                    ptr += offset;
                } else {
                    ptr = memory.len();
                    memory.resize(memory.len().checked_add(1).unwrap(), 0);
                }
            }).to_tokens(tokens);
        } else {
            (quote! {
                ptr = ::esolangs::brainfuck::macro_reexports::memchr::memrchr(#value, &(&memory)[..=ptr]).unwrap();
            }).to_tokens(tokens);
        }
    }
}

// `+`, `-`, and derivatives thereof.
// Subtraction is implemented with wrapping addition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct IncrementCell {
    amount: u8,
    offset: PtrOffset,
}

impl ToTokens for IncrementCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let amount = self.amount;
        let offset = self.offset;
        if amount != 0 {
            (quote! {
                memory[ptr #offset] = memory[ptr #offset].wrapping_add(#amount);
            })
            .to_tokens(tokens);
        }
    }
}

// `[-]`, `[-]+`, etc.
// Also produced by multiply lopps
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct SetCell {
    value: u8,
    offset: PtrOffset,
}

impl ToTokens for SetCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let value = self.value;
        let offset = self.offset;
        (quote! {
            memory[ptr #offset] = #value;
        })
        .to_tokens(tokens);
    }
}

// Produced by multiply loops like `[->++<]`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct MultiplyToCell {
    source_offset: PtrOffset,
    target_offset: PtrOffset,
    coefficient: u8,
}

impl ToTokens for MultiplyToCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let source_offset = self.source_offset;
        let target_offset = self.target_offset;
        let coef = self.coefficient;
        (match coef {
            0 => quote! { },
            1 => quote! { memory[ptr #target_offset] = memory[ptr #source_offset].wrapping_add(memory[ptr #target_offset]) },
            coef => quote! { memory[ptr #target_offset] = (memory[ptr #source_offset] * #coef).wrapping_add(memory[ptr #target_offset]) },
        })
        .to_tokens(tokens);
    }
}

// `,`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ReadToCell {
    offset: PtrOffset,
}

impl ToTokens for ReadToCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let offset = self.offset;

        (quote! {
            if ::std::io::Read::read_exact(&mut input, &mut buf).is_ok() {
                memory[ptr #offset] = buf[0];
            }
        })
        .to_tokens(tokens);
    }
}

// `;`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct PrintCell {
    offset: PtrOffset,
}

impl ToTokens for PrintCell {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let offset = self.offset;

        (quote! {
            if let Err(err) = ::std::io::Write::write(&mut output, &[memory[ptr #offset]]) {
                if err.kind() == ::std::io::ErrorKind::Interrupted {
                    let _ = ::std::io::Write::write(&mut output, &[memory[ptr #offset]]);
                }
            }
            let _ = ::std::io::Write::flush(&mut output);
        })
        .to_tokens(tokens);
    }
}

// `;` when the value to be printed is known at compile time
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

// `[stuff]`
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

// `#`
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

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Ast {
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
    fn parse_bytes_inner(instructions: &mut Vec<Instruction>, bytes: &[u8]) {
        let mut i: usize = 0;

        while i < bytes.len() {
            match bytes[i] {
                b'>' => {
                    instructions.push(Instruction::ShiftPtr(ShiftPtr {
                        offset: PtrOffset::Forward(1),
                    }));
                    i += 1;
                }
                b'<' => {
                    instructions.push(Instruction::ShiftPtr(ShiftPtr {
                        offset: PtrOffset::Backward(1),
                    }));
                    i += 1;
                }
                b'+' => {
                    instructions.push(Instruction::IncrementCell(IncrementCell {
                        amount: 1,
                        offset: PtrOffset::Zero,
                    }));
                    i += 1;
                }
                b'-' => {
                    instructions.push(Instruction::IncrementCell(IncrementCell {
                        amount: 255,
                        offset: PtrOffset::Zero,
                    }));
                    i += 1;
                }
                b'.' => {
                    instructions.push(Instruction::PrintCell(PrintCell {
                        offset: PtrOffset::Zero,
                    }));
                    i += 1;
                }
                b',' => {
                    instructions.push(Instruction::ReadToCell(ReadToCell {
                        offset: PtrOffset::Zero,
                    }));
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
                    let mut inner_vec = Vec::with_capacity(inner_bytes.len());
                    parse_bytes_inner(&mut inner_vec, &inner_bytes);
                    instructions.push(Instruction::Loop(Loop { inner: inner_vec }));
                }
                _ => i += 1,
            }
        }
    }

    let mut instructions: Vec<Instruction> = Vec::with_capacity(bytes.len() + 1);
    instructions.push(Instruction::ProgramStart(ProgramStart {}));

    parse_bytes_inner(&mut instructions, bytes);

    instructions
}

// Enum for instructions thar read from or write to a cell
#[derive(PartialEq, Eq, Debug)]
enum InstrAccessingCell<'a> {
    ProgramStart(&'a mut ProgramStart),
    IncrementCell(&'a mut IncrementCell),
    SetCell(&'a mut SetCell),
    MultiplyToCell(&'a mut MultiplyToCell),
    ReadToCell(&'a mut ReadToCell),
    PrintCell(&'a mut PrintCell),
    Loop(&'a mut Loop),
    ScanLoop(&'a mut ScanLoop),
}

macro_rules! impl_from_instr_accessing {
    ($($instr_ty:ident),*) => {$(
        impl<'a> From<&'a mut $instr_ty> for InstrAccessingCell<'a> {
            fn from(instr: &'a mut $instr_ty) -> Self {
                InstrAccessingCell::$instr_ty(instr)
            }
        }
    )*};
}

impl_from_instr_accessing!(IncrementCell, SetCell, ReadToCell, PrintCell, Loop);

// Used during optimization: when adding each new instruction to the optimize output,
// we first search for whether there is a possible optimization involving a previous operation involving
// the same cell.
fn last_instr_accessing_cell(
    prev_instrs: &mut [Instruction],
    mut offset: PtrOffset,
) -> Option<(usize, InstrAccessingCell)> {
    let mut last_accessing: Option<(usize, InstrAccessingCell)> = None;

    for prev_instr in prev_instrs.iter_mut().enumerate().rev() {
        //println!("            Offset: {:?}", offset);
        match prev_instr {
            (_, Instruction::ShiftPtr(shift)) => offset -= shift.offset,
            (i, Instruction::ProgramStart(start)) => {
                last_accessing = Some((i, InstrAccessingCell::ProgramStart(start)));
                assert_eq!(i, 0);
                break;
            }
            (_, Instruction::Debug(_) | Instruction::PrintValue(_)) => {
                last_accessing = None;
                break;
            }
            (i, Instruction::IncrementCell(incr)) => {
                if offset == incr.offset {
                    last_accessing = Some((i, InstrAccessingCell::IncrementCell(incr)));
                    break;
                }
            }
            (i, Instruction::SetCell(set)) => {
                if offset == set.offset {
                    last_accessing = Some((i, InstrAccessingCell::SetCell(set)));
                    break;
                }
            }
            (i, Instruction::MultiplyToCell(mul)) => {
                if offset == mul.source_offset || offset == mul.target_offset {
                    last_accessing = Some((i, InstrAccessingCell::MultiplyToCell(mul)));
                    break;
                }
            }
            (i, Instruction::ReadToCell(read)) => {
                if offset == read.offset {
                    last_accessing = Some((i, InstrAccessingCell::ReadToCell(read)));
                    break;
                }
            }
            (i, Instruction::PrintCell(print)) => {
                if offset == print.offset {
                    last_accessing = Some((i, InstrAccessingCell::PrintCell(print)));
                    break;
                }
            }
            (i, Instruction::Loop(loop_instr)) => {
                if offset == PtrOffset::Zero {
                    last_accessing = Some((i, InstrAccessingCell::Loop(loop_instr)));
                } else {
                    last_accessing = None;
                    // TODO check loop interior?
                }
                break;
            }
            (i, Instruction::ScanLoop(scan)) => {
                if offset == PtrOffset::Zero {
                    last_accessing = Some((i, InstrAccessingCell::ScanLoop(scan)));
                } else {
                    last_accessing = None;
                }
                break;
            }
        }
    }

    //println!("        Last accessing: {:?}", last_accessing);
    last_accessing
}

fn last_coalescable_ptr_shift_idx(
    prev_instrs: &mut [Instruction],
) -> Option<(usize, &mut ShiftPtr)> {
    //println!("Checking last shift");

    let mut shift_idx = None;
    for prev_instr in prev_instrs.iter_mut().enumerate().rev() {
        match prev_instr {
            (idx, Instruction::ShiftPtr(shift)) => {
                shift_idx = Some((idx, shift));
                break;
            }
            (_, Instruction::PrintValue(_)) => {}
            _ => {
                shift_idx = None;
                break;
            }
        };
    }
    //println!("Last shift {:?}", shift_idx);

    shift_idx
}

fn optimize_coalescing_pass(input: &[Instruction]) -> Vec<Instruction> {
    fn optimized_add_instr(output: &mut Vec<Instruction>, instr: Instruction) {
        //println!("    {:?} {:?} Adding {:?}", output.len(), output, instr);
        match instr {
            Instruction::ProgramStart(_) => {
                assert_eq!(output.len(), 0);
                output.push(instr);
            }
            Instruction::IncrementCell(incr) => {
                // TODO switch to @ bindings on Rust 1.56
                let amount = incr.amount;
                let offset = incr.offset;
                if let Some((idx, prev_instr)) = last_instr_accessing_cell(output, incr.offset) {
                    match prev_instr {
                        InstrAccessingCell::ProgramStart(_) => {
                            output.push(Instruction::SetCell(SetCell {
                                value: amount,
                                offset,
                            }));
                        }
                        InstrAccessingCell::IncrementCell(prev_incr) => {
                            prev_incr.amount = prev_incr.amount.wrapping_add(amount);
                            if prev_incr.amount == 0 {
                                output.remove(idx);
                            }
                        }
                        InstrAccessingCell::SetCell(prev_set) => {
                            prev_set.value = prev_set.value.wrapping_add(amount);
                        }
                        InstrAccessingCell::Loop(_) => output.push(Instruction::SetCell(SetCell {
                            value: amount,
                            offset,
                        })),
                        InstrAccessingCell::ScanLoop(scan) => {
                            let value = scan.value;
                            output.push(Instruction::SetCell(SetCell {
                                value: amount + value,
                                offset,
                            }))
                        }
                        _ => output.push(instr),
                    }
                } else {
                    output.push(instr);
                }
            }
            Instruction::SetCell(set) => {
                let mut prev_value: Option<u8> = None;
                while let Some((idx, prev_instr)) = last_instr_accessing_cell(output, set.offset) {
                    match prev_instr {
                        InstrAccessingCell::IncrementCell(_)
                        | InstrAccessingCell::MultiplyToCell(_)
                        | InstrAccessingCell::SetCell(_) => {
                            output.remove(idx);
                        }
                        InstrAccessingCell::ProgramStart(_) | InstrAccessingCell::Loop(_) => {
                            prev_value = Some(0);
                            break;
                        }
                        InstrAccessingCell::ScanLoop(scan) => {
                            let value = scan.value;
                            prev_value = Some(value);
                            break;
                        }
                        _ => break,
                    }
                }
                match prev_value {
                    Some(val) if val == set.value => {}
                    _ => output.push(instr),
                }
            }
            Instruction::MultiplyToCell(_) => output.push(instr), //TODO
            Instruction::PrintCell(print) => {
                match last_instr_accessing_cell(output, print.offset) {
                    Some((_, InstrAccessingCell::SetCell(set))) => {
                        let value = set.value;
                        output.push(Instruction::PrintValue(PrintValue { value }));
                    }
                    Some((
                        _,
                        InstrAccessingCell::ProgramStart(_) | InstrAccessingCell::Loop(_),
                    )) => {
                        output.push(Instruction::PrintValue(PrintValue { value: 0 }));
                    }
                    Some((_, InstrAccessingCell::ScanLoop(scan))) => {
                        let value = scan.value;
                        output.push(Instruction::PrintValue(PrintValue { value }));
                    }
                    _ => output.push(instr),
                }
            }
            Instruction::ShiftPtr(shift) => {
                if let Some((_, prev_shift)) = last_coalescable_ptr_shift_idx(output) {
                    prev_shift.offset += shift.offset;
                } else {
                    output.push(instr);
                }
            }
            Instruction::Loop(mut loop_instr) => {
                loop_instr.inner = optimize_instructions(&loop_instr.inner);

                let mut is_loop = true;
                if loop_instr.inner.len() == 1 {
                    match loop_instr.inner[0] {
                        Instruction::IncrementCell(incr) => {
                            if incr.amount == 0 {
                                loop_instr.inner.clear();
                            } else {
                                is_loop = false;
                                optimized_add_instr(
                                    output,
                                    Instruction::SetCell(SetCell {
                                        value: 0,
                                        offset: PtrOffset::Zero,
                                    }),
                                );
                            }
                        }
                        Instruction::ShiftPtr(shift) => match shift.offset {
                            // TODO detect more kinds of scan loops
                            PtrOffset::Backward(1usize) => {
                                is_loop = false;
                                optimized_add_instr(
                                    output,
                                    Instruction::ScanLoop(ScanLoop {
                                        value: 0,
                                        forward: false,
                                    }),
                                );
                            }
                            PtrOffset::Forward(1usize) => {
                                is_loop = false;
                                optimized_add_instr(
                                    output,
                                    Instruction::ScanLoop(ScanLoop {
                                        value: 0,
                                        forward: true,
                                    }),
                                );
                            }
                            _ => {}
                        },
                        _ => (),
                    }
                }
                if is_loop {
                    match last_instr_accessing_cell(output, PtrOffset::Zero) {
                        Some((_, prev_instr)) => match prev_instr {
                            InstrAccessingCell::SetCell(set) => {
                                if set.value != 0 {
                                    output.push(Instruction::Loop(loop_instr));
                                }
                            }
                            InstrAccessingCell::Loop(_)
                            | InstrAccessingCell::ProgramStart(_)
                            | InstrAccessingCell::ScanLoop(ScanLoop { value: 0, .. }) => {}
                            _ => output.push(Instruction::Loop(loop_instr)),
                        },
                        _ => output.push(Instruction::Loop(loop_instr)),
                    }
                }
            }
            Instruction::ScanLoop(scan) => {
                match last_instr_accessing_cell(output, PtrOffset::Zero) {
                    Some((_, prev_instr)) => match prev_instr {
                        InstrAccessingCell::SetCell(SetCell { value, .. })
                        | InstrAccessingCell::ScanLoop(ScanLoop { value, .. })
                            if value == &scan.value => {}
                        InstrAccessingCell::Loop(_) | InstrAccessingCell::ProgramStart(_)
                            if 0 == scan.value => {}
                        _ => output.push(instr),
                    },
                    _ => output.push(instr),
                }
            }
            _ => output.push(instr),
        }
    }

    //println!("Optimize");
    //println!(" Input: {:?}", input);

    let mut output: Vec<Instruction> = Vec::with_capacity(input.len());

    for instr in input {
        optimized_add_instr(&mut output, instr.clone());
    }

    output
}

fn optimize_reordering_pass() {
    todo!();
}

fn optimize_instructions(input: &[Instruction]) -> Vec<Instruction> {
    let mut pre = input.to_vec();
    let mut post: Vec<Instruction>;

    loop {
        post = optimize_coalescing_pass(&pre);
        if post == pre {
            break;
        }

        pre = post;
    }

    post
}

impl ToTokens for Ast {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut instr_tokens = TokenStream::new();
        for instr in &self.instructions {
            instr.to_tokens(&mut instr_tokens);
        }

        (quote! {
            {
                #instr_tokens
            }
        })
        .to_tokens(tokens);
    }
}
