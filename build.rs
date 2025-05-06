fn main() {
    // 设置环境变量
    println!("cargo:rustc-env=TORCH_LIBRARY_PATH=/usr/local/lib");
    println!("cargo:rustc-env=LD_LIBRARY_PATH=/usr/local/lib");
}
