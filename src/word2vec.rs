use bincode;
use memmap::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Lines, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

/// Metadata
#[derive(Debug)]
pub struct Word2VecMeta {
    pub len: usize,
    pub dim: usize,
}

/// Trait for a read-only store of Word2Vec embeddings.
pub trait Word2VecStore: Sized {
    /// Load the store into memory.
    fn load(path: &str) -> Result<Self, Box<Error>>;
    /// Fetch the vector representation of a word.
    fn get(&self, key: &str) -> Option<Vec<f32>>;
    /// Get the size of the store.
    fn len(&self) -> usize;
    /// Get the number of dimensions.
    fn dim(&self) -> usize;
}

/// Memory-mapped interface to a set of Word2Vec embeddings.
#[derive(Debug)]
pub struct Word2VecMmap {
    index: HashMap<String, usize>,
    store: Mmap,
    meta: Word2VecMeta,
}

impl Word2VecMmap {
    /// Load a Word2Vec
    pub fn load_from<P: AsRef<Path>>(idx_file: P, vec_file: P) -> Result<Self, Box<Error>> {
        let store = File::open(vec_file.as_ref())?;
        let store = unsafe { MmapOptions::new().map(&store)? };
        let (meta, index) = {
            let file = File::open(idx_file.as_ref())?;
            let file = BufReader::with_capacity(128 * 1024 * 1024, file);
            let mut lines = file.lines();
            let meta = parse_header_from_lines(&mut lines)?;
            let index = lines
                .into_iter()
                .map(|line| {
                    let line = line.unwrap();
                    let idx = line.find(' ').unwrap();
                    (
                        line[..idx].to_string(),
                        line[idx + 1..].parse::<usize>().unwrap(),
                    )
                })
                .collect::<HashMap<String, usize>>();
            (meta, index)
        };
        Ok(Word2VecMmap { index, store, meta })
    }
}

impl Word2VecStore for Word2VecMmap {
    fn load(path: &str) -> Result<Self, Box<Error>> {
        let idx_file = format!("{}.idx", path);
        let vec_file = format!("{}.vec", path);
        Word2VecMmap::load_from(idx_file, vec_file)
    }
    fn len(&self) -> usize {
        self.meta.len
    }

    fn dim(&self) -> usize {
        self.meta.dim
    }

    fn get(&self, key: &str) -> Option<Vec<f32>> {
        if let Some(offset) = self.index.get(key) {
            match bincode::deserialize(&self.store[*offset..]) {
                Ok(data) => return Some(data),
                Err(err) => {
                    println!("{:?}", err);
                }
            }
        }
        None
    }
}

/// Consume the header line from a Word2Vec text file or .idx file, returning parsed metadata.
fn parse_header_from_lines<B: BufRead>(lines: &mut Lines<B>) -> Result<Word2VecMeta, Box<Error>> {
    let head = lines
        .next()
        .transpose()?
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Header not found"))?;

    let idx = head.find(' ').unwrap();
    let len = head[..idx].parse::<usize>()?;
    let dim = head[idx + 1..].parse::<usize>()?;

    Ok(Word2VecMeta { len, dim })
}

/// Read a plaintext w2v file to memory.
pub(crate) fn read_path_to_memory<P: AsRef<Path>>(
    path: P,
    buf_size: Option<usize>,
) -> Result<(Word2VecMeta, Vec<String>), Box<Error>> {
    let file = File::open(path.as_ref())?;
    let buf_size = buf_size.unwrap_or(1024 * 1024 * 1024);
    let reader = BufReader::with_capacity(buf_size, file);
    read_to_memory(reader)
}

/// Read a plaintext w2v file to memory.
pub(crate) fn read_to_memory<R: BufRead>(
    reader: R,
) -> Result<(Word2VecMeta, Vec<String>), Box<Error>> {
    let mut lines = reader.lines();

    let meta = parse_header_from_lines(&mut lines)?;

    let lines = lines
        .enumerate()
        .map(|(idx, line)| {
            if idx % 100000 == 0 {
                println!("Done {} lines", idx);
            }
            line.unwrap()
        })
        .collect::<Vec<_>>();
    Ok((meta, lines))
}

/// Serialise a .txt Word2Vec embeddings file to an (.idx, .vec) pair for use in a memory-mapped Word2VecMmap store.
pub fn serialize_to_store<P: AsRef<Path>>(path: P) -> Result<(PathBuf, PathBuf), Box<Error>> {
    let stem = path
        .as_ref()
        .file_stem()
        .map_or("w2v_store", |stem| stem.to_str().unwrap_or("w2v_store"));
    let idx_mtx_file = format!("{}.idx", stem);
    let vec_mtx_file = format!("{}.vec", stem);

    println!("Ingesting...");
    let (meta, lines) = read_path_to_memory(path, None)?;
    println!("Done!");

    println!("Transforming...");

    let idx_mtx = File::create(&idx_mtx_file)?;
    let idx_mtx = BufWriter::with_capacity(128 * 1024 * 1024, idx_mtx);
    let idx_mtx = Mutex::new(idx_mtx);

    let vec_mtx = File::create(&vec_mtx_file)?;
    let vec_mtx = BufWriter::with_capacity(1024 * 1024 * 1024, vec_mtx);
    let vec_mtx = Mutex::new(vec_mtx);

    lines
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
        .map(|chunk| {
            let mut vec_file = vec_mtx.lock().unwrap();
            let chunk = chunk
                .into_iter()
                .map(|(key, value)| {
                    let offset = vec_file.seek(SeekFrom::Current(0)).unwrap();
                    vec_file.write(&value).unwrap();
                    (key, offset)
                })
                .collect::<Vec<_>>();
            println!("Wrote batch");
            chunk
        })
        .for_each(|index_chunk| {
            let mut idx_file = idx_mtx.lock().unwrap();
            writeln!(idx_file, "{} {}", meta.len, meta.dim).unwrap();
            for (key, offset) in index_chunk {
                writeln!(idx_file, "{} {}", key, offset).unwrap();
            }
        });
    println!("Done!");
    Ok((PathBuf::from(idx_mtx_file), PathBuf::from(vec_mtx_file)))
}
