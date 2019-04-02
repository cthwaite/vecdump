///
///
/// fn main() -> Result<(), Box<Error>> {
///     let (idx, vec) = serialize_to_store("../../Documents/enwiki_20180420_300d.txt")?;
///     let vec = Word2VecMmap::load(&idx, &vec).unwrap();
///     let v = vec.get("the");
///     println!("{:?}", v);
///     Ok(())
/// }
///
pub mod rocks;
pub mod word2vec;
