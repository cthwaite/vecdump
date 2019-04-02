#![cfg(feature = "rocksdb")]
use crate::word2vec::read_to_memory;
use bincode;
use rayon::prelude::*;
use rocksdb::{WriteBatch, DB as RocksDB};
use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

pub fn serialize_to_rocksdb<P: AsRef<Path>>(path: P) -> Result<PathBuf, Box<Error>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::with_capacity(1024 * 1024 * 1024, file);
    let stem = path
        .as_ref()
        .file_stem()
        .map_or("w2v_store", |stem| stem.to_str().unwrap_or("w2v_store"));
    let db_path = format!("{}.db", stem);

    println!("Loading...");
    let mut chunk_counter = 0;
    let (_meta, lines) = read_to_memory(reader)?;
    println!("Done!");

    println!("Transforming...");

    let db_mtx = Mutex::new(RocksDB::open_default(db_path.clone())?);

    let transformed = lines
        .into_par_iter()
        .map(|line| {
            let idx = line.find(' ').unwrap();
            let label = &line[..idx];
            let vec = line[idx + 1..]
                .split_whitespace()
                .map(|v| lexical::parse::<f32, _>(v))
                .collect::<Vec<_>>();
            (label.to_string(), bincode::serialize(&vec).unwrap())
        })
        .chunks(100000)
        .for_each(|chunk| {
            let mut batch = WriteBatch::default();
            for (key, value) in chunk {
                batch.put(key, value);
            }
            {
                let db = db_mtx.lock().unwrap();
                db.write(batch);
            }
            println!("Wrote batch");
        });
    println!("Done!");
    Ok(PathBuf::from(db_path))
}
