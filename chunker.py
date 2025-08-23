import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class DocumentChunk:
    """Represents a chunk of Indonesian legal document content with metadata"""
    level: int  # 1-3
    title: str
    content: str
    section_number: Optional[str] = None
    parent_section: Optional[str] = None
    chunk_type: Optional[str] = None  # metadata, bab, pasal. 
    start_position: int = 0
    end_position: int = 0
    chunk_id: str = ""
    metadata: Dict[str, Any] = None

class IndonesianLegalDocumentChunker:
    """
    Chunks Indonesian legal documents into 5 hierarchical levels:
    Level 1: Metadata (Judul, Nomor, Tahun, Tentang, Menimbang, Mengingat, Status)
    Level 2: Major Sections (Pasal groups, Lampiran sections)
    Level 3: Pasal (Articles)
    """
    
    def __init__(self, max_chunk_size: int = 1500, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.document_metadata = {}
        
        # Regex patterns for Indonesian legal documents
        self.patterns = {
            # Level 1: Metadata patterns
            'metadata': {
                'judul': r'^(PERATURAN\s+(?:MENTERI|PEMERINTAH|PRESIDEN|DAERAH)[^\n]+?)(?=\s+NOMOR|\s+TAHUN|$)',
                'nomor': r'NOMOR\s+([^\n]+)',
                'tahun': r'TAHUN\s+(\d{4})',
                'tentang': r'TENTANG\s*\n(.+?)(?=\nMenimbang|\nMengingat|MEMUTUSKAN:|$)',
                'menimbang': r'Menimbang\s*:\s*(.*?)(?=Mengingat\s*:|MEMUTUSKAN:)',
                'mengingat': r'Mengingat\s*:\s*(.*?)(?=MEMUTUSKAN:|$)',
                'memutuskan': r'MEMUTUSKAN:\s*\n',
                'menetapkan': r'Menetapkan\s*:\s*(.+?)(?=\n\n|Pasal\s+\d+)'
            },
            
            # Level 2: Major sections
            'major_sections': [
                r'^(LAMPIRAN\s+[IVX]+)\s*$',        # Lampiran
                r'^(BAB\s+[IVX]+)\s*',              # Bab I, Bab II
                r'^(BAGIAN\s+[A-Z]+)\s*',           # Bagian Kesatu
            #    r'^[A-Z]\.\s+[A-Z][A-Z\s]+$'        # A. PENDAHULUAN (all caps only)
            ],

            
            # Level 3: Pasal (Articles)
            'pasal': [
                r'^(Pasal\s+\d+)\s*$',
                r'^(Pasal\s+\d+[a-z]?)\s*$'
            ],
        }
    
    def read_document(self, file_path: str) -> str:
        """Read document from file with multiple encoding support"""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not read file {file_path} with any supported encoding")
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract Level 1 metadata from the document header"""
        metadata = {}

            # --- JUDUL (full title) ---
        judul_match = re.search(
                r'^(PERATURAN\s+(?:MENTERI|PEMERINTAH|PRESIDEN|DAERAH)[^\n]+?)(?=\s+NOMOR|\s+TAHUN|$)',
                text, re.MULTILINE | re.IGNORECASE
            )
        if judul_match:
                metadata['judul'] = judul_match.group(1).strip()

            # --- NOMOR (number + year in one line) ---
        nomor_match = re.search(
                r'NOMOR\s+(\d+\s+TAHUN\s+\d{4})',
                text, re.IGNORECASE
            )
        if nomor_match:
                metadata['nomor'] = nomor_match.group(1).strip()

            # --- TAHUN (separate if needed) ---
        tahun_match = re.search(r'TAHUN\s+(\d{4})', text, re.IGNORECASE)
        if tahun_match:
                metadata['tahun'] = tahun_match.group(1).strip()

            # --- TENTANG (stop at Menimbang/Mengingat/Memutuskan) ---
        tentang_match = re.search(
                r'TENTANG\s*\n(.+?)(?=\nMenimbang|\nMengingat|MEMUTUSKAN:|$)',
                text, re.MULTILINE | re.DOTALL | re.IGNORECASE
            )
        if tentang_match:
                metadata['tentang'] = tentang_match.group(1).strip().replace('\n', ' ')

            # --- MENIMBANG (lettered list: a., b., c.) ---
        menimbang_match = re.search(
                r'Menimbang\s*:\s*(.*?)(?=\n\s*Mengingat\s*:|MEMUTUSKAN:)',
                text, re.MULTILINE | re.DOTALL | re.IGNORECASE
            )
        if menimbang_match:
            content = menimbang_match.group(1).strip()
            items = re.findall(r'([a-z])\.\s*(.+?)(?=(?:[a-z]\.\s|$))', content, re.DOTALL)
            if items:
                metadata['menimbang'] = [
                    {'point': point, 'text': txt.strip().replace('\n', ' ')}
                    for point, txt in items
                ]
            else:
                # fallback: whole paragraph as one point
                metadata['menimbang'] = [{'point': 'a', 'text': content.replace('\n', ' ')}]

        # --- MENGINGAT ---
        # 1) find exact block boundaries using simple searches
        start_m = re.search(r'Mengingat\s*:', text, re.IGNORECASE)
        if start_m:
            after_start = text[start_m.end():]
            end_m = re.search(r'\n\s*MEMUTUSKAN\s*:', after_start, re.IGNORECASE)
            block = after_start[:end_m.start()] if end_m else after_start

            # 2) normalize newlines/spaces (handle CRLF and non-breaking spaces)
            block = block.replace('\r\n', '\n').replace('\r', '\n').replace('\u00A0', ' ')
            # Some PDFs can introduce weird spacing; collapse repeated spaces on each line when joining later

            # 3) split on numbered items at line start (1., 2., ...)
            parts = re.split(r'(?m)^\s*(\d+)\.\s*', block)

            items = []
            # parts[0] is preface (usually empty); then pairs of (num, text)
            for i in range(1, len(parts), 2):
                num = parts[i]
                txt = parts[i+1] if i + 1 < len(parts) else ""
                # Trim and collapse whitespace over the whole item
                txt = re.sub(r'\s+', ' ', txt).strip(' ;')
                if txt:
                    items.append({"point": num, "text": txt})

            # 4) assign
            metadata['mengingat'] = items


            # --- MEMUTUSKAN ---
        memutuskan_match = re.search(r'MEMUTUSKAN:', text, re.IGNORECASE)
        if memutuskan_match:
                metadata['memutuskan'] = "MEMUTUSKAN:"

            # --- MENETAPKAN ---
        menetapkan_match = re.search(
                r'Menetapkan\s*:\s*(.+?)(?=\n\n|Pasal\s+\d+)',
                text, re.MULTILINE | re.DOTALL | re.IGNORECASE
            )
        if menetapkan_match:
                metadata['menetapkan'] = menetapkan_match.group(1).strip().replace('\n', ' ')

            # --- Document type heuristic ---
        metadata['document_type'] = 'Peraturan Menteri' if 'MENTERI' in text else 'Peraturan'

            # --- Signing Information ---
        signing_match = re.search(
                r'Ditetapkan di (.+?)\s+pada tanggal (.+?)\s+(.+?),\s*ttd\s+(.+?)(?=\n)',
                text, re.MULTILINE | re.DOTALL
            )
        if signing_match:
                metadata['tempat_penetapan'] = signing_match.group(1).strip()
                metadata['tanggal_penetapan'] = signing_match.group(2).strip()
                metadata['jabatan_penandatangan'] = signing_match.group(3).strip()
                metadata['nama_penandatangan'] = signing_match.group(4).strip()
        return metadata

    
    def identify_structure_level(self, line: str) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        """
        Identify the structural level of a line
        Returns: (level, section_number, title, chunk_type)
        """
        line = line.strip()
        if not line:
            return 0, None, None, None
        
        # Level 2: Major sections
        for pattern in self.patterns['major_sections']:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1) if match.lastindex else match.group(0)
                return 2, None, title, 'major_section'
        
        # Level 3: Pasal
        for pattern in self.patterns['pasal']:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return 3, match.group(1), match.group(1), 'pasal'
        
        return 0, None, None, None  # Regular content line
    
    def split_into_sections(self, text: str) -> List[Dict]:
        """Split document into hierarchical sections"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            level, section_num, title, chunk_type = self.identify_structure_level(line)
            
            if level > 0:  # This is a structural element
                # Save previous section if exists
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    current_section['end_position'] = i - 1
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'level': level,
                    'title': title or line.strip(),
                    'section_number': section_num,
                    'chunk_type': chunk_type,
                    'start_position': i,
                    'raw_line': line
                }
                current_content = []
            # --- Default: add content to current section ---
            else:
                if line.strip():
                    current_content.append(line)

        # Add the last section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            current_section['end_position'] = len(lines) - 1
            sections.append(current_section)
        
        return sections
    
    def establish_hierarchy(self, sections: List[Dict]) -> List[Dict]:
        """Establish parent-child relationships between sections"""
        hierarchy_stack = []
        
        for section in sections:
            level = section['level']
            
            # Pop sections from stack that are at same or higher level
            while hierarchy_stack and hierarchy_stack[-1]['level'] >= level:
                hierarchy_stack.pop()
            
            # Set parent relationship
            if hierarchy_stack:
                parent = hierarchy_stack[-1]
                section['parent_section'] = parent.get('section_number') or parent.get('title')
                section['parent_title'] = parent['title']
                section['parent_type'] = parent.get('chunk_type')
            else:
                section['parent_section'] = None
                section['parent_title'] = None
                section['parent_type'] = None
            
            hierarchy_stack.append(section)
        
        return sections
    
    def chunk_long_content(self, content: str, base_chunk: Dict) -> List[DocumentChunk]:
        """Split long content into smaller chunks while preserving context"""
        if len(content) <= self.max_chunk_size:
            return [DocumentChunk(
                level=base_chunk['level'],
                title=base_chunk['title'],
                content=content,
                section_number=base_chunk['section_number'],
                parent_section=base_chunk.get('parent_section'),
                chunk_type=base_chunk.get('chunk_type'),
                start_position=base_chunk['start_position'],
                end_position=base_chunk['end_position'],
                metadata=self.document_metadata
            )]
        
        chunks = []
        # Split by sentences while preserving Indonesian punctuation
        sentences = re.split(r'(?<=[.!?;])\s+', content)
        
        current_chunk = ""
        chunk_count = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_count += 1
                chunk_id = f"{base_chunk.get('section_number', 'chunk')}_{chunk_count}"
                
                chunks.append(DocumentChunk(
                    level=base_chunk['level'],
                    title=f"{base_chunk['title']} (Bagian {chunk_count})",
                    content=current_chunk.strip(),
                    section_number=base_chunk['section_number'],
                    parent_section=base_chunk.get('parent_section'),
                    chunk_type=base_chunk.get('chunk_type'),
                    chunk_id=chunk_id,
                    metadata=self.document_metadata
                ))
                
                # Start new chunk with overlap
                if self.overlap_size > 0:
                    overlap_sentences = current_chunk.split('. ')[-2:]
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_count += 1
            chunk_id = f"{base_chunk.get('section_number', 'chunk')}_{chunk_count}"
            
            chunks.append(DocumentChunk(
                level=base_chunk['level'],
                title=f"{base_chunk['title']} (Bagian {chunk_count})" if chunk_count > 1 else base_chunk['title'],
                content=current_chunk.strip(),
                section_number=base_chunk['section_number'],
                parent_section=base_chunk.get('parent_section'),
                chunk_type=base_chunk.get('chunk_type'),
                chunk_id=chunk_id,
                metadata=self.document_metadata
            ))
        
        return chunks
    
    def create_metadata_chunk(self) -> DocumentChunk:
        """Create Level 1 metadata chunk"""
        metadata_content = []
        
        if self.document_metadata.get('judul'):
            metadata_content.append(f"Judul: {self.document_metadata['judul']}")
        if self.document_metadata.get('nomor'):
            metadata_content.append(f"Nomor: {self.document_metadata['nomor']}")
        if self.document_metadata.get('tahun'):
            metadata_content.append(f"Tahun: {self.document_metadata['tahun']}")
        if self.document_metadata.get('tentang'):
            metadata_content.append(f"Tentang: {self.document_metadata['tentang']}")
        
        # Add menimbang items
        if self.document_metadata.get('menimbang'):
            metadata_content.append("\nMenimbang:")
            for item in self.document_metadata['menimbang']:
                metadata_content.append(f"  {item['point']}. {item['text']}")
        
        # Add mengingat items
        if self.document_metadata.get('mengingat'):
            metadata_content.append("\nMengingat:")
            for item in self.document_metadata['mengingat']:
                metadata_content.append(f"  {item['point']}. {item['text']}")
        
        return DocumentChunk(
            level=1,
            title="Metadata Dokumen",
            content='\n'.join(metadata_content),
            chunk_type='metadata',
            chunk_id='metadata',
            metadata=self.document_metadata
        )
    
    def chunk_document(self, text: str) -> List[DocumentChunk]:
        """Main method to chunk Indonesian legal document into 3 levels"""
        # Extract metadata first
        self.document_metadata = self.extract_metadata(text)
        
        # Create metadata chunk (Level 1)
        all_chunks = [self.create_metadata_chunk()]
        
        # Split into sections (Levels 2-3)
        sections = self.split_into_sections(text)
        
        # Establish hierarchy
        sections = self.establish_hierarchy(sections)
        
        # Convert to DocumentChunk objects
        for section in sections:
            if section['content']:  # Only process sections with content
                section_chunks = self.chunk_long_content(section['content'], section)
                all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def chunk_file(self, file_path: str) -> List[DocumentChunk]:
        """Chunk an Indonesian legal document file"""
        text = self.read_document(file_path)
        return self.chunk_document(text)
    
    def export_chunks_to_dict(self, chunks: List[DocumentChunk]) -> List[Dict]:
        """Export chunks to dictionary format for easy serialization"""
        return [
            {
                'level': chunk.level,
                'title': chunk.title,
                'content': chunk.content,
                'section_number': chunk.section_number,
                'parent_section': chunk.parent_section,
                'chunk_type': chunk.chunk_type,
                'chunk_id': chunk.chunk_id,
                'content_length': len(chunk.content),
                'metadata': chunk.metadata if chunk.level == 1 else None
            }
            for chunk in chunks
        ]
    
    def export_to_json(self, chunks: List[DocumentChunk], file_path: str):
        """Export chunks to JSON file"""
        data = self.export_chunks_to_dict(chunks)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def print_chunk_summary(self, chunks: List[DocumentChunk]):
        """Print a summary of the chunking results"""
        level_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        type_counts = {}
        total_chars = 0
        
        for chunk in chunks:
            level_counts[chunk.level] += 1
            chunk_type = chunk.chunk_type or 'unknown'
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            total_chars += len(chunk.content)