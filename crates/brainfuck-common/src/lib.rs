use proc_macro2::TokenStream;
use quote::ToTokens;

mod ast;
pub fn brainfuck2(item: TokenStream) -> TokenStream {
    let item_str: &str = &item.to_string();
    println!("Str: {}", item_str);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let bytes = item_str.as_bytes();
    brainfuck_from_bytes(bytes)
}

fn brainfuck_from_bytes(bytes: &[u8]) -> TokenStream {
    let mut ast = ast::Ast::from_bytes(bytes);
    ast.optimize();
    ast.to_token_stream()
}

pub fn brainfuck_include2(item: TokenStream) -> TokenStream {
    let lit_str: syn::LitStr = syn::parse2(item).expect("Failed to  parse bf filename");
    let mut path = std::env::current_dir().expect("Failed to access cwd");
    path.push(lit_str.value());
    let path_str = format!("{:?}", path);
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|_| panic!("Failed to open BF file {}", path_str));
    let mut bytes = Vec::<u8>::new();
    std::io::Read::read_to_end(&mut file, &mut bytes)
        .unwrap_or_else(|_| panic!("Failed to read BF file {}", path_str));
    brainfuck_from_bytes(&bytes)
}
