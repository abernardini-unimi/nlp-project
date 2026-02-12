from pathlib import Path
import asyncio
import pickle
import gzip

from config.logger import logger

from src.text_processor import read_doc_file, read_docx_file, read_pdf_file, read_text_file

 
async def save_compressed_pickle(filepath: Path, data) -> bool:
    try:
        await asyncio.to_thread(
            lambda: gzip.open(filepath, 'wb').write(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
        )
        return True
    except Exception as e:
        logger.error(f"❌ Error saving file {filepath}: {e}")
        return False


async def load_compressed_pickle(filepath: Path):
    try:
        data = await asyncio.to_thread(
            lambda: pickle.loads(gzip.open(filepath, 'rb').read())
        )
        return data
    except Exception as e:
        logger.error(f"❌ Error loading file {filepath}: {e}")
        return None


async def load_text_from_file(filepath: str) -> str:
    """Load text from a file based on its extension."""
    file_path = Path(filepath)
    ext = file_path.suffix.lower()

    match ext:
        case ".pdf":
            return await asyncio.to_thread(read_pdf_file, file_path)
        case ".docx":
            return await asyncio.to_thread(read_docx_file, file_path)
        case ".doc":
            return await asyncio.to_thread(read_doc_file, file_path)
        case _:
            return await asyncio.to_thread(read_text_file, file_path)