pub mod brainfuck {
    pub use brainfuck_macro::{brainfuck, brainfuck_include};

    #[doc(hidden)]
    pub mod macro_reexports {
        pub use ::memchr;
        pub use ::tinyvec;
    }
}
