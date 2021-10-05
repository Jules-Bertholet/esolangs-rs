use proc_macro::TokenStream;

#[proc_macro]
pub fn brainfuck(item: TokenStream) -> TokenStream {
    brainfuck_common::brainfuck2(item.into()).into()
}

#[proc_macro]
pub fn brainfuck_include(item: TokenStream) -> TokenStream {
    brainfuck_common::brainfuck_include2(item.into()).into()
}
