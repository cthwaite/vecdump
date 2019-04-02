use std::env;
use std::error::Error;
#[cfg(feature = "rocks-db")]
use vecdump::rocks::serialize_to_rocksdb;
use vecdump::word2vec::serialize_to_store;

/// Dump .txt to .db
#[cfg(feature = "rocks-db")]
fn main() -> Result<(), Box<Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("USAGE: dump [PATH]");
        return Ok(());
    }
    let path = args[1].clone();
    println!("Loading {}...", path);
    let db = serialize_to_rocksdb(&path)?;
    println!("Wrote db to {:?}", db);
    println!("Done!");
    Ok(())
}

/// Dump .txt to (.idx, .vec)
fn main() -> Result<(), Box<Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("USAGE: dump [PATH]");
        return Ok(());
    }
    let path = args[1].clone();
    println!("Loading {}...", path);
    let (idx, vec) = serialize_to_store(&path)?;
    println!("Wrote index to {:?}", idx);
    println!("Wrote vectors to {:?}", vec);
    println!("Done!");
    Ok(())
}
