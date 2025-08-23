import requests
import time
import json

# Test document content (sample Indonesian regulation)
test_document = """
PERATURAN MENTERI LINGKUNGAN HIDUP DAN KEHUTANAN
REPUBLIK INDONESIA
NOMOR 14 TAHUN 2024
TENTANG
PENYELENGGARAAN PENGAWASAN DAN SANKSI ADMINISTRATIF
DI BIDANG LINGKUNGAN HIDUP

DENGAN RAHMAT TUHAN YANG MAHA ESA

MENTERI LINGKUNGAN HIDUP DAN KEHUTANAN REPUBLIK INDONESIA,

Menimbang:
a. bahwa untuk menjamin pelaksanaan peraturan perundang-undangan di bidang lingkungan hidup, perlu dilakukan pengawasan yang efektif;
b. bahwa pengawasan di bidang lingkungan hidup perlu didukung dengan sanksi administratif yang tegas dan proporsional;
c. bahwa berdasarkan pertimbangan sebagaimana dimaksud dalam huruf a dan huruf b, perlu menetapkan Peraturan Menteri Lingkungan Hidup dan Kehutanan tentang Penyelenggaraan Pengawasan dan Sanksi Administratif di Bidang Lingkungan Hidup;

Mengingat:
1. Undang-Undang Nomor 32 Tahun 2009 tentang Perlindungan dan Pengelolaan Lingkungan Hidup;
2. Peraturan Pemerintah Nomor 22 Tahun 2021 tentang Penyelenggaraan Perlindungan dan Pengelolaan Lingkungan Hidup;

MEMUTUSKAN:

Menetapkan: PERATURAN MENTERI LINGKUNGAN HIDUP DAN KEHUTANAN TENTANG PENYELENGGARAAN PENGAWASAN DAN SANKSI ADMINISTRATIF DI BIDANG LINGKUNGAN HIDUP.

BAB I
KETENTUAN UMUM

Pasal 1
Dalam Peraturan Menteri ini yang dimaksud dengan:
1. Lingkungan hidup adalah kesatuan ruang dengan semua benda, daya, keadaan, dan makhluk hidup, termasuk manusia dan perilakunya, yang mempengaruhi alam itu sendiri, kelangsungan perikehidupan, dan kesejahteraan manusia serta makhluk hidup lain.
2. Pengawasan lingkungan hidup adalah serangkaian kegiatan pengamatan terhadap ketaatan penanggung jawab usaha dan/atau kegiatan atas ketentuan yang ditetapkan dalam peraturan perundang-undangan.

Pasal 2
Pengawasan lingkungan hidup bertujuan:
a. menjamin ketaatan terhadap peraturan perundang-undangan di bidang lingkungan hidup;
b. menjamin terlaksananya perlindungan dan pengelolaan lingkungan hidup.

BAB II
PENGAWASAN

Pasal 3
Pengawasan lingkungan hidup dilaksanakan oleh:
a. Menteri;
b. gubernur; dan
c. bupati/walikota.
"""

def test_chunker_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Indonesian Legal Document Chunker API")
    print("=" * 60)
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test document chunking
    print("\n2. Starting document chunking...")
    try:
        chunk_request = {
            "text": test_document,
            "options": {
                "max_chunk_size": 1500,
                "overlap_size": 100
            }
        }
        
        response = requests.post(f"{base_url}/chunk-document", json=chunk_request)
        print(f"✅ Chunk request: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result["job_id"]
            print(f"   Job ID: {job_id}")
            
            # Poll for progress
            print("\n3. Monitoring progress...")
            while True:
                progress_response = requests.get(f"{base_url}/job/{job_id}/progress")
                if progress_response.status_code == 200:
                    progress = progress_response.json()
                    print(f"   📊 {progress['stage']}: {progress['progress']:.1f}% - {progress['message']}")
                    
                    if progress['stage'] in ['completed', 'failed']:
                        break
                    
                    time.sleep(2)
                else:
                    print(f"   ❌ Progress check failed: {progress_response.status_code}")
                    break
            
            # Get final result
            print("\n4. Getting final result...")
            result_response = requests.get(f"{base_url}/job/{job_id}/result")
            if result_response.status_code == 200:
                final_result = result_response.json()
                print(f"✅ Processing completed!")
                print(f"   Status: {final_result['status']}")
                print(f"   Processing time: {final_result.get('processing_time', 0):.2f}s")
                
                if final_result.get('summary'):
                    summary = final_result['summary']
                    print(f"   Total chunks: {summary['total_chunks']}")
                    print(f"   Level counts: {summary['level_counts']}")
                    print(f"   Type counts: {summary['type_counts']}")
                    print(f"   Total characters: {summary['total_characters']:,}")
                
                if final_result.get('metadata'):
                    metadata = final_result['metadata']
                    print(f"\n📋 Extracted Metadata:")
                    print(f"   Judul: {metadata.get('judul', 'N/A')}")
                    print(f"   Nomor: {metadata.get('nomor', 'N/A')}")
                    print(f"   Tahun: {metadata.get('tahun', 'N/A')}")
                    print(f"   Tentang: {metadata.get('tentang', 'N/A')[:100]}...")
                
                # Show sample chunks
                if final_result.get('chunks'):
                    chunks = final_result['chunks']
                    print(f"\n📄 Sample Chunks:")
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                        print(f"   Chunk {i+1}: Level {chunk['level']} - {chunk['title']}")
                        print(f"   Content: {chunk['content'][:100]}...")
                        print(f"   Type: {chunk['chunk_type']}, Length: {chunk['content_length']}")
                        print()
                
            else:
                print(f"❌ Failed to get result: {result_response.status_code}")
                
        else:
            print(f"❌ Chunk request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_chunker_api()