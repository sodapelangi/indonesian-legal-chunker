import re
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum
from abc import ABC, abstractmethod

class DocumentType(Enum):
    PERATURAN_PEMERINTAH = "Peraturan Pemerintah"
    PERATURAN_MENTERI = "Peraturan Menteri"
    PERATURAN_PRESIDEN = "Peraturan Presiden"
    PERATURAN_DAERAH = "Peraturan Daerah"
    UNDANG_UNDANG = "Undang-Undang"
    KEPUTUSAN = "Keputusan"
    INSTRUKSI = "Instruksi"
    SURAT_EDARAN = "Surat Edaran"
    UNKNOWN = "Unknown"

@dataclass
class DocumentChunk:
    """Represents a chunk of Indonesian legal document content with metadata"""
    level: int
    title: str
    content: str
    section_number: Optional[str] = None
    parent_section: Optional[str] = None
    chunk_type: Optional[str] = None
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_type: DocumentType = DocumentType.UNKNOWN

class DocumentTypeDetector:
    """Enhanced detector for Indonesian legal documents with MD support"""
    
    def __init__(self):
        # Fixed patterns with markdown awareness
        self.type_patterns = {
            DocumentType.PERATURAN_PEMERINTAH: [
                r'^PERATURAN\s+PEMERINTAH\s+REPUBLIK\s+INDONESIA',
                r'^PERATURAN\s+PEMERINTAH\s+NOMOR',
                r'PP\s+NOMOR\s+\d+\s+TAHUN\s+\d{4}'
            ],
            DocumentType.PERATURAN_MENTERI: [
                r'^PERATURAN\s+MENTERI\s+[A-Z][A-Z\s]+(?:NOMOR|DAN|REPUBLIK)',
                r'MENTERI\s+[A-Z][A-Z\s]+\s+REPUBLIK\s+INDONESIA',
                r'^PERATURAN\s+MENTERI\s+NOMOR',
                r'PERMEN\s+NOMOR'
            ],
            DocumentType.UNDANG_UNDANG: [
                r'^UNDANG-UNDANG\s+REPUBLIK\s+INDONESIA',
                r'^UNDANG-UNDANG\s+NOMOR',
                r'UU\s+NOMOR\s+\d+\s+TAHUN\s+\d{4}'
            ],
            DocumentType.PERATURAN_PRESIDEN: [
                r'^PERATURAN\s+PRESIDEN\s+REPUBLIK\s+INDONESIA',
                r'^PERATURAN\s+PRESIDEN\s+NOMOR',
                r'PERPRES\s+NOMOR'
            ],
            DocumentType.PERATURAN_DAERAH: [
                r'^PERATURAN\s+DAERAH\s+[A-Z\s]+\s+NOMOR',
                r'PERDA\s+NOMOR'
            ],
            DocumentType.KEPUTUSAN: [
                r'KEPUTUSAN\s+[A-Z\s]+\s+NOMOR',
                r'KEPUTUSAN\s+PRESIDEN\s+NOMOR',
                r'KEPUTUSAN\s+MENTERI\s+NOMOR'
            ],
            DocumentType.INSTRUKSI: [
                r'INSTRUKSI\s+[A-Z\s]+\s+NOMOR',
                r'INSTRUKSI\s+PRESIDEN\s+NOMOR',
                r'INSTRUKSI\s+MENTERI\s+NOMOR'
            ],
            DocumentType.SURAT_EDARAN: [
                r'SURAT\s+EDARAN\s+[A-Z\s]+\s+NOMOR',
                r'SURAT\s+EDARAN\s+NOMOR'
            ]
        }

    def preprocess_for_detection(self, text: str) -> str:
        """Preprocess text for better document type detection"""
        # Remove markdown formatting
        text = re.sub(r'\[.*?\]', '', text)  # Remove markdown links
        text = re.sub(r'!\[.*?\].*?\)', '', text)  # Remove image references
        text = re.sub(r'#+\s*', '', text)  # Remove markdown headers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Remove italic formatting
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.upper()
    

    def detect_document_type(self, text: str) -> Tuple[DocumentType, float]:
        search_text = text[:1000].upper()
        for doc_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, search_text): return doc_type, 1.0
        return DocumentType.UNKNOWN, 0.0
    
class BasePatternExtractor(ABC):
    @abstractmethod
    def extract_metadata(self, text: str) -> Dict[str, Any]: 
        pass
    @abstractmethod
    def identify_structure_level(self, line: str) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        pass

class EnhancedLegalPatternExtractor(BasePatternExtractor):
    """Enhanced pattern extractor with better MD support"""
    
    def __init__(self):
        self.patterns = {
            'metadata': {
                'judul_full': r'((?:PERATURAN|UNDANG-UNDANG|KEPUTUSAN|INSTRUKSI|SURAT\s+EDARAN)[^\n]*?)(?=\s*NOMOR|\s*$)',
                # CORRECTED PATTERN:
                'nomor_tahun': r'NOMOR\s+([^\n]*?)\s+TAHUN\s+(\d{4})',
                'tentang': r'TENTANG\s+((?:.|\n)+?)(?=\s*DENGAN RAHMAT|Menimbang|Mengingat|$)',
                'menimbang': r'Menimbang\s*:?\s*((?:.|\n)+?)(?=Mengingat|MEMUTUSKAN:|$)',
                'mengingat': r'Mengingat\s*:?\s*((?:.|\n)+?)(?=MEMUTUSKAN:|Dengan\s+Persetujuan\s+Bersama|$)',
                'menetapkan': r'MEMUTUSKAN\s*:\s*\n*\s*Menetapkan\s*:\s*([^\n]+)',
                'penetapan': r'(?:Ditetapkan|Disahkan)\s+di\s+([^\n]+?)\s+pada\s+tanggal\s+([\d]{1,2}\s+\w+\s+\d{4})',
                'pengundangan': r'Diundangkan\s+di\s+([^\n]+?)\s+pada\s+tanggal\s+([\d]{1,2}\s+\w+\s+\d{4})',
        }}
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        metadata = {}
        judul_match = re.search(self.patterns['metadata']['judul_full'], text, re.MULTILINE)
        if judul_match: metadata['judul'] = ' '.join(judul_match.group(0).strip().split())
        
        nomor_tahun_match = re.search(self.patterns['metadata']['nomor_tahun'], text, re.IGNORECASE)
        if nomor_tahun_match:
            metadata['nomor'] = f"{nomor_tahun_match.group(1).strip()} TAHUN {nomor_tahun_match.group(2)}"
            metadata['tahun'] = nomor_tahun_match.group(2)
            
        tentang_match = re.search(self.patterns['metadata']['tentang'], text, re.IGNORECASE | re.DOTALL)
        if tentang_match:
            tentang_text = ' '.join(tentang_match.group(1).strip().split())
            metadata['tentang'] = tentang_text

        self._extract_menimbang(text, metadata)
        self._extract_mengingat(text, metadata)
        
        menetapkan_match = re.search(self.patterns['metadata']['menetapkan'], text)
        if menetapkan_match: metadata['menetapkan'] = menetapkan_match.group(1).strip().replace('\n', ' ')
        self._extract_signing_info(text, metadata)
        self._extract_promulgation_info(text, metadata)
        return metadata
    
    def _parse_list_items(self, text_block: str, pattern: str) -> List[Dict[str, str]]:
        items, current_point, current_text = [], None, []
        lines = text_block.strip().split('\n')
        if not lines: return []
        if not any(re.match(pattern, line.strip(), re.IGNORECASE) for line in lines):
            clean_text = ' '.join(line.strip() for line in lines if line.strip())
            return [{'point': 'a', 'text': re.sub(r'\s+', ' ', clean_text).strip()}]
        
        for line in lines:
            line_content = line.strip()
            if not line_content: continue
            
            # --- THE FIX: PERFORM THE REGEX MATCH HERE ---
            # This line was missing. It creates the `match` variable.
            match = re.match(pattern, line_content, re.IGNORECASE)
            if match:
                # If we were tracking a previous item, save it now
                if current_point: 
                    items.append({'point': current_point, 'text': re.sub(r'\s+', ' ', ' '.join(current_text)).strip()})
                
                # Start the new item
                current_point = match.group(1)
                current_text = [line_content[match.end():].strip()]
            elif current_point:
                # This line is a continuation of the current item
                current_text.append(line_content)
            
        if current_point: items.append({'point': current_point, 'text': re.sub(r'\s+', ' ', ' '.join(current_text)).strip()})
        return items
    
    def _extract_menimbang(self, text: str, metadata: Dict[str, Any]):
        menimbang_match = re.search(self.patterns['metadata']['menimbang'], text, re.IGNORECASE)
        if menimbang_match: metadata['menimbang'] = self._parse_list_items(menimbang_match.group(1), r'^\s*([a-z])\.\s*')

    def _extract_mengingat(self, text: str, metadata: Dict[str, Any]):
        mengingat_match = re.search(self.patterns['metadata']['mengingat'], text, re.IGNORECASE)
        if mengingat_match: metadata['mengingat'] = self._parse_list_items(mengingat_match.group(1), r'^\s*(\d+)\.\s*')

    def _extract_signing_info(self, text: str, metadata: Dict[str, Any]):
        match = re.search(self.patterns['metadata']['penetapan'], text, re.MULTILINE | re.IGNORECASE)
        if match:
            metadata['tempat_penetapan'] = match.group(1).strip().rstrip(',.:')
            metadata['tanggal_penetapan'] = match.group(2).strip().rstrip(',.:')

    def _extract_promulgation_info(self, text: str, metadata: Dict[str, Any]):
        match = re.search(self.patterns['metadata']['pengundangan'], text, re.MULTILINE | re.IGNORECASE)
        if match:
            metadata['tempat_pengundangan'] = match.group(1).strip().rstrip(',.:')
            metadata['tanggal_pengundangan'] = match.group(2).strip().rstrip(',.:')

    def identify_structure_level(self, line: str) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        """
        Identifies the structural level of a line from a document.

        Levels:
        - 1: Meta data, content text, or "Penjelasan".
        - 2: Lampiran, Bab, Bagian, or a Roman Numeral section.
        - 3: Pasal (Article).
        - 0: Empty line.

        Returns:
            A tuple (level, identifier, title, type).
            - level (int): The identified structure level (0-3).
            - identifier (str | None): A unique key for the section (e.g., "Pasal 1", "BAB I").
            - title (str | None): The full text of the line.
            - type (str | None): A category name (e.g., 'pasal', 'bab', 'metadata').
        """
        line = line.strip()
        if not line:
            return 0, None, None, None # Level 0 for empty lines

        # Level 3: Pasal (most specific, so check first)
        # Matches "Pasal 1", "Pasal 2A", "Pasal X", etc.
        pasal_match = re.match(r'^(Pasal\s+[\w\d]+)', line, re.IGNORECASE)
        if pasal_match:
            identifier = pasal_match.group(0).strip() # e.g., "Pasal 1"
            title = line
            return 3, identifier, title, 'pasal'

        # Level 2: Lampiran, Bab, Bagian, Roman Numeral sections
        level2_patterns = [
            # Matches "LAMPIRAN", "LAMPIRAN I", "LAMPIRAN KEPUTUSAN..."
            (r'^(LAMPIRAN.*)', 'lampiran'),
            # Matches "BAB I", "BAB II", etc.
            (r'^(BAB\s+[IVXLCDM]+)', 'bab'),
            # Matches "BAGIAN KESATU", "BAGIAN KEDUA", etc.
            (r'^(BAGIAN\s+K[A-Z]+)', 'bagian'),
            # Matches a standalone Roman numeral, e.g., "I.", "II.", which often denotes a major section
            (r'^([IVXLCDM]+)\.', 'roman_numeral_section')
        ]
        
        for pattern, type_name in level2_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Capture the matched part as the identifier (e.g., "BAB I", "LAMPIRAN")
                identifier = match.group(1).strip()
                title = line
                return 2, identifier, title, type_name

        # Level 1: Meta data and Penjelasan
        # If the line is not a Level 2 or 3 structure, it defaults to Level 1.
        
        # We can specifically identify "PENJELASAN" to give it a unique type
        if re.match(r'^PENJELASAN$', line, re.IGNORECASE):
            return 1, 'PENJELASAN', 'PENJELASAN', 'penjelasan'
        
        # All other non-empty lines are considered metadata or general content.
        return 1, None, line, 'metadata_or_content'
    
class AdaptiveIndonesianLegalDocumentChunker:
    def __init__(self, max_chunk_size: int = 2500, overlap_size: int = 150):
        self.max_chunk_size, self.overlap_size = max_chunk_size, overlap_size
        self.document_metadata, self.document_type = {}, DocumentType.UNKNOWN
        self.type_detector, self.extractor = DocumentTypeDetector(), EnhancedLegalPatternExtractor()
        # FINAL FIX for FIND_REG_PATTERN: More specific and robust
        self.FIND_REG_PATTERN = r"((?:(?:Keputusan|Peraturan)\s+(?:Menteri|Presiden|Pemerintah|Menteri\s+Negara|Daerah)|Undang-Undang)\s+Nomor\s+[\w\s/]+\s+Tahun\s+\d{4}\s+tentang\s(?:.|\n)*?)(?=\s*(?:;|\n\s*[a-z]\.|\(Berita|\)$))"

    def read_document(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file: return file.read()

    # In the AdaptiveIndonesianLegalDocumentChunker class

    def _analyze_regulation_status(self, full_text: str) -> Dict[str, str]:
        """
        Analyzes the text by searching for a definitive revocation first.
        If no revocation is found, it then checks for amendments.
        """
        # --- 1. DEFINE PATTERNS (The most important part) ---
        preamble_end_match = re.search(r'MEMUTUSKAN:', full_text)
        preamble_text = full_text[:preamble_end_match.start()] if preamble_end_match else full_text[:5000]

        penutup_text = ""
        penutup_match = re.search(r'BAB\s+[IVXLCDM]+\s+KETENTUAN\s+PENUTUP', full_text, re.IGNORECASE)
        if penutup_match:
            penutup_text = full_text[penutup_match.start():]
        else: # Fallback
            penutup_text = full_text[-5000:]
        
        # Pattern to check for amendments in the preamble (used as a fallback).
        # FIXED PATTERN
        revocation_context_pattern = re.compile(r'((?:.|\n)*?)dicabut\s+dan\s+dinyatakan\s+tidak\s+berlaku', re.IGNORECASE)
        amend_pattern = re.compile(r"(?:sebagaimana\s+telah\s+diubah\s+dengan|PERUBAHAN\s+(?:.*?)\s+ATAS)\s+((?:PERATURAN|UNDANG-UNDANG|KEPUTUSAN)[\s\S]*?)(?=\s*(?:DENGAN RAHMAT|Menimbang|Mengingat|$))", 
            re.IGNORECASE | re.DOTALL)
        find_all_regs_pattern = re.compile(self.FIND_REG_PATTERN, re.IGNORECASE)
        
        # --- 2. PRIMARY ACTION: CHECK FOR REVOCATION ---
        # We search the entire document for the revocation context.
        context_match = revocation_context_pattern.search(penutup_text)
        if context_match:
            revocation_context = context_match.group(1)
            found_regulations = find_all_regs_pattern.findall(revocation_context)
            if found_regulations:
                cleaned_regulations = [re.sub(r'\s+', ' ', reg).strip().rstrip(';') for reg in found_regulations]
                return {"status": "Mencabut (Revokes)", "target_regulation": "\n".join(cleaned_regulations)}

        # --- 3. SECONDARY ACTION: CHECK FOR AMENDMENT (ONLY IF NO REVOCATION WAS FOUND) ---
        # We only need to search the preamble for this.
        #preamble_end_match = re.search(r'MEMUTUSKAN:', full_text, re.IGNORECASE)
        #preamble_text = full_text[:preamble_end_match.start()] if preamble_end_match else full_text[:5000]
        
        match = amend_pattern.search(preamble_text)
        if match:
            return {
                "status": "Mengubah (Amends)",
                "target_regulation": re.sub(r'\s+', ' ', match.group(1).strip())
            }

        # --- 4. DEFAULT (if neither was found) ---
        return {"status": "Tidak Mengubah/Mencabut (Does Not Amend/Revoke)", "target_regulation": "N/A"}
    
    def chunk_document(self, text: str) -> List[DocumentChunk]:
        self.document_type, confidence = self.type_detector.detect_document_type(text)
        self.document_metadata = self.extractor.extract_metadata(text)
        self.document_metadata['document_type'] = self.document_type.value
        self.document_metadata['detection_confidence'] = confidence
        status_info = self._analyze_regulation_status(text)
        self.document_metadata.update(status_info)
        all_chunks = [self.create_metadata_chunk()]
        sections = self.split_into_sections(text)
        sections = self.establish_hierarchy(sections)
        for section in sections:
            if section.get('content'):
                section_chunks = self.chunk_long_content(section['content'], section)
                all_chunks.extend(section_chunks)   

        return all_chunks
                
    def create_metadata_chunk(self) -> DocumentChunk:
        metadata_content = []
        meta = self.document_metadata
        if meta.get('judul'): metadata_content.append(f"Judul: {meta['judul']}")
        if meta.get('nomor'): metadata_content.append(f"Nomor: {meta['nomor']}")
        if meta.get('tahun'): metadata_content.append(f"Tahun: {meta['tahun']}")
        if meta.get('tentang'): metadata_content.append(f"Tentang: {meta['tentang']}")
        if meta.get('status') and meta['status'] != 'Tidak Mengubah/Mencabut (Does Not Amend/Revoke)':
            metadata_content.append(f"\nStatus Regulasi: {meta['status']}")
            metadata_content.append(f"Regulasi Terdampak:\n{meta['target_regulation']}")
        if meta.get('menimbang'):
            metadata_content.append("\nMenimbang:")
            for item in meta['menimbang']: metadata_content.append(f"  {item['point']}. {item['text']}")
        if meta.get('mengingat'):
            metadata_content.append("\nMengingat:")
            for item in meta['mengingat']: metadata_content.append(f"  {item['point']}. {item['text']}")
        if meta.get('tempat_penetapan') or meta.get('tanggal_penetapan'):
             metadata_content.append("\n" + f"Tempat Penetapan: {meta.get('tempat_penetapan', 'N/A')}")
             metadata_content.append(f"Tanggal Penetapan: {meta.get('tanggal_penetapan', 'N/A')}")
        if meta.get('tempat_pengundangan') or meta.get('tanggal_pengundangan'):
             metadata_content.append(f"Tempat Pengundangan: {meta.get('tempat_pengundangan', 'N/A')}")
             metadata_content.append(f"Tanggal Pengundangan: {meta.get('tanggal_pengundangan', 'N/A')}")
        return DocumentChunk(
            level=1, title="Metadata Dokumen", content='\n'.join(metadata_content),
            chunk_type='metadata', chunk_id='metadata', metadata=self.document_metadata,
            document_type=self.document_type)
    
    def split_into_sections(self, text: str) -> List[Dict]:
        """
        Splits a legal text document into structured sections based on
        preamble markers and structural level identification.

        Args:
            text (str): The full text content of the document.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents
                        a parsed section with its level, title, section_number,
                        chunk_type, and content.
        """
        lines = text.split('\n')
        sections: List[Dict] = []
        current_section: Optional[Dict] = None
        current_content: List[str] = []
        start_index = 0

        # Step 1: Find preamble end (robustly and case-insensitively)
        text_upper = text.upper()
        memutuskan_pos = text_upper.find("MEMUTUSKAN:")
        menetapkan_pos = text_upper.find("MENETAPKAN:")
        start_pos = -1 # Character index of the preamble keyword
        
        if memutuskan_pos != -1 and menetapkan_pos != -1:
            start_pos = min(memutuskan_pos, menetapkan_pos)
        elif memutuskan_pos != -1:
            start_pos = memutuskan_pos
        elif menetapkan_pos != -1:
            start_pos = menetapkan_pos

        # Step 2: Find the real content start ("BAB I" or "Pasal 1") using Regex
        # Only search for content start if a preamble marker was found
        if start_pos != -1:
            # Look for structural markers only AFTER the preamble
            post_preamble_text = text[start_pos:]
            # Using regex for robustness: 'BAB I' or 'Pasal I' or 'Pasal 1'
            match = re.search(r'(BAB\s+I|Pasal\s+(?:I\b|1\b))', post_preamble_text, re.IGNORECASE)
            
            if match:
                # Calculate the line number of the match within the original text
                start_index = text[:start_pos].count('\n') + post_preamble_text[:match.start()].count('\n')
            else:
                # If no BAB I/Pasal 1 found, start parsing from the line after the preamble marker
                start_index = text[:start_pos].count('\n') + 1
        # If no preamble marker found, start_index remains 0, meaning parse from beginning

        # Step 3: Main loop to split the text into sections
        for i, line in enumerate(lines[start_index:], start=start_index):
            clean_line = line.strip()
            if not clean_line:
                # If there's a current section, empty lines are part of its content
                if current_section:
                    current_content.append("") # Preserve empty lines within content
                continue

            # Use the external extractor to classify the line
            level, section_num, title, chunk_type = self.extractor.identify_structure_level(clean_line)

            if level > 1: # This line is a new significant section (Level 2: BAB/Lampiran, Level 3: Pasal)
                if current_section and current_content:
                    # Save the previously collected section
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start a new section
                current_section = {
                    'level': level,
                    'title': title,
                    'section_number': section_num,
                    'chunk_type': chunk_type
                }
                current_content = []
                
                # Check if there's content on the same line as the title (e.g., "Pasal 1: Text content...")
                line_content_after_title = clean_line[len(title):].strip()
                if line_content_after_title:
                    current_content.append(line_content_after_title)
            elif current_section: # This line is content for the current section (including Level 1 content)
                current_content.append(clean_line)
            # If level is 1 and current_section is None, it means we are encountering metadata/content
            # before any actual structural section (BAB/Pasal). These lines are ignored
            # as they are implicitly part of the preamble or introductory text we're skipping.

        # Step 4: Save the very last section after the loop ends
        if current_section and current_content:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def establish_hierarchy(self, sections: List[Dict]) -> List[Dict]:
        hierarchy_stack = []
        for section in sections:
            level = section.get('level', 0)
            while hierarchy_stack and hierarchy_stack[-1].get('level', 0) >= level: hierarchy_stack.pop()
            section['parent_section'] = hierarchy_stack[-1].get('section_number') or hierarchy_stack[-1].get('title') if hierarchy_stack else None
            if section.get('chunk_type') != 'pasal': hierarchy_stack.append(section)
        return sections

    def chunk_long_content(self, content: str, base_chunk: Dict) -> List[DocumentChunk]:
        if len(content) <= self.max_chunk_size:
            return [DocumentChunk(level=base_chunk['level'], title=base_chunk['title'], content=content,
                                  section_number=base_chunk.get('section_number'), parent_section=base_chunk.get('parent_section'),
                                  chunk_type=base_chunk.get('chunk_type'), metadata=self.document_metadata, document_type=self.document_type)]
        chunks, part = [], 1
        sentences = re.split(r'(?<=[.!?;])\s+', content)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    level=base_chunk['level'], title=f"{base_chunk['title']} (Bagian {part})", content=current_chunk.strip(),
                    section_number=base_chunk.get('section_number'), parent_section=base_chunk.get('parent_section'),
                    chunk_type=base_chunk.get('chunk_type'), chunk_id=f"{base_chunk.get('section_number', 'chunk')}_{part}",
                    metadata=self.document_metadata, document_type=self.document_type))
                part += 1; current_chunk = sentence
            else: current_chunk += (' ' + sentence) if current_chunk else sentence
        if current_chunk:
            chunks.append(DocumentChunk(
                level=base_chunk['level'], title=f"{base_chunk['title']} (Bagian {part})" if part > 1 else base_chunk['title'], content=current_chunk.strip(),
                section_number=base_chunk.get('section_number'), parent_section=base_chunk.get('parent_section'),
                chunk_type=base_chunk.get('chunk_type'), chunk_id=f"{base_chunk.get('section_number', 'chunk')}_{part}",
                metadata=self.document_metadata, document_type=self.document_type))
        return chunks

    def chunk_file(self, file_path: str) -> List[DocumentChunk]:
        text = self.read_document(file_path); return self.chunk_document(text)

    def export_chunks_to_dict(self, chunks: List[DocumentChunk]) -> List[Dict]:
        return [{'level': c.level, 'title': c.title, 'content': c.content, 'section_number': c.section_number,
                 'parent_section': c.parent_section, 'chunk_type': c.chunk_type, 'chunk_id': c.chunk_id,
                 'content_length': len(c.content), 'metadata': c.metadata if c.chunk_type == 'metadata' else None}
                for c in chunks]

    def export_to_json(self, chunks: List[DocumentChunk], file_path: str):
        data = self.export_chunks_to_dict(chunks)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
# Add this line to the bottom of your existing chunker.py file
# This creates an alias so your app.py can find the class it expects

IndonesianLegalDocumentChunker = AdaptiveIndonesianLegalDocumentChunker
