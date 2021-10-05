pub mod brainfuck {
    pub mod prelude {
        pub use brainfuck_macro::{brainfuck, brainfuck_include};
        pub use tinyvec::tiny_vec as __bf_macro_tiny_vec;
    }
}
