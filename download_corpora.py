#!/usr/bin/env python3

import requests
import json
from pathlib import Path
import gzip
import urllib.request

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def extract_gzip(gz_file, txt_file):
    """Extract gzipped file."""
    try:
        with gzip.open(gz_file, 'rt', encoding='utf-8') as f_in:
            with open(txt_file, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        print(f"✓ Extracted {txt_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract {gz_file}: {e}")
        return False

def create_sample_corpus():
    """Create a sample corpus with diverse text types."""
    sample_texts = [
        # Academic/Technical
        "The quantum algorithm demonstrates exponential speedup over classical methods.",
        "Machine learning models require substantial computational resources for training.",
        "The neural network architecture consists of multiple convolutional layers.",
        
        # News/Journalism
        "The government announced new economic policies to address inflation concerns.",
        "Scientists discovered evidence of ancient civilizations in remote regions.",
        "The company reported record profits despite market volatility.",
        
        # Fiction/Creative
        "The old wizard cast a powerful spell that illuminated the dark chamber.",
        "She gazed at the stars, wondering about the mysteries of the universe.",
        "The detective carefully examined the evidence at the crime scene.",
        
        # Technical Documentation
        "Install the software package using the command line interface.",
        "Configure the network settings to enable remote access.",
        "The API requires authentication tokens for secure communication.",
        
        # Social Media Style
        "Just finished reading an amazing book about space exploration!",
        "Can't believe how fast technology is advancing these days.",
        "The weather is perfect for a weekend hiking trip.",
        
        # Business/Professional
        "The quarterly report indicates strong growth in international markets.",
        "Our team successfully completed the project ahead of schedule.",
        "The conference featured presentations from industry experts.",
        
        # Educational
        "Students learn fundamental concepts through hands-on experiments.",
        "The textbook explains complex theories using clear examples.",
        "Research shows that active learning improves retention rates.",
        
        # Scientific
        "The experiment confirmed the hypothesis about molecular interactions.",
        "Data analysis revealed significant correlations between variables.",
        "The study population included participants from diverse backgrounds."
    ]
    
    corpus_file = "data/corpora/sample_diverse.txt"
    Path(corpus_file).write_text("\n".join(sample_texts), encoding="utf-8")
    print(f"✓ Created {corpus_file} with {len(sample_texts)} diverse texts")
    return corpus_file

def main():
    """Download and prepare diverse corpora."""
    print("🌐 Downloading Diverse Corpora for Primitive Testing")
    print("=" * 60)
    
    # Create corpora directory
    Path("data/corpora").mkdir(exist_ok=True)
    
    # Create sample diverse corpus
    sample_corpus = create_sample_corpus()
    
    # Try to download some real corpora (with fallbacks)
    corpora = [
        {
            "name": "news_headlines",
            "url": "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/data/legitimate/legitimate.csv",
            "filename": "data/corpora/news_headlines.csv",
            "description": "News headlines and articles"
        },
        {
            "name": "academic_abstracts", 
            "url": "https://raw.githubusercontent.com/allenai/s2orc/master/data/abstracts_sample.txt",
            "filename": "data/corpora/academic_abstracts.txt",
            "description": "Academic paper abstracts"
        }
    ]
    
    downloaded_corpora = [sample_corpus]
    
    for corpus in corpora:
        if download_file(corpus["url"], corpus["filename"]):
            downloaded_corpora.append(corpus["filename"])
    
    print(f"\n📊 Summary:")
    print(f"Successfully prepared {len(downloaded_corpora)} corpora:")
    for corpus in downloaded_corpora:
        print(f"  - {corpus}")
    
    return downloaded_corpora

if __name__ == "__main__":
    main()

