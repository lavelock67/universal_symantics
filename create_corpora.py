#!/usr/bin/env python3

from pathlib import Path

def create_news_corpus():
    """Create a news corpus."""
    news_texts = [
        "The government announced new economic policies to address inflation concerns.",
        "Scientists discovered evidence of ancient civilizations in remote regions.",
        "The company reported record profits despite market volatility.",
        "International leaders met to discuss climate change initiatives.",
        "The stock market experienced significant fluctuations during trading hours.",
        "Researchers published findings on breakthrough medical treatments.",
        "The election results showed a clear majority for the incumbent party.",
        "Technology companies invested heavily in artificial intelligence research.",
        "Environmental activists protested against industrial pollution.",
        "The education system implemented new digital learning platforms."
    ]
    
    corpus_file = "data/corpora/news_corpus.txt"
    Path(corpus_file).write_text("\n".join(news_texts), encoding="utf-8")
    print(f"✓ Created {corpus_file} with {len(news_texts)} news texts")
    return corpus_file

def create_academic_corpus():
    """Create an academic corpus."""
    academic_texts = [
        "The quantum algorithm demonstrates exponential speedup over classical methods.",
        "Machine learning models require substantial computational resources for training.",
        "The neural network architecture consists of multiple convolutional layers.",
        "Statistical analysis revealed significant correlations between variables.",
        "The experimental results confirmed the theoretical predictions.",
        "Researchers conducted longitudinal studies on population dynamics.",
        "The methodology employed rigorous peer review processes.",
        "Data collection involved systematic sampling techniques.",
        "The literature review identified gaps in current research.",
        "Hypothesis testing utilized advanced statistical methods."
    ]
    
    corpus_file = "data/corpora/academic_corpus.txt"
    Path(corpus_file).write_text("\n".join(academic_texts), encoding="utf-8")
    print(f"✓ Created {corpus_file} with {len(academic_texts)} academic texts")
    return corpus_file

def create_technical_corpus():
    """Create a technical documentation corpus."""
    technical_texts = [
        "Install the software package using the command line interface.",
        "Configure the network settings to enable remote access.",
        "The API requires authentication tokens for secure communication.",
        "Database queries must include proper indexing for optimal performance.",
        "The deployment process involves multiple staging environments.",
        "Error handling mechanisms prevent system crashes during execution.",
        "The configuration file specifies connection parameters.",
        "Memory allocation requires careful management to avoid leaks.",
        "The build system compiles source code into executable binaries.",
        "Logging mechanisms track application behavior for debugging."
    ]
    
    corpus_file = "data/corpora/technical_corpus.txt"
    Path(corpus_file).write_text("\n".join(technical_texts), encoding="utf-8")
    print(f"✓ Created {corpus_file} with {len(technical_texts)} technical texts")
    return corpus_file

def create_fiction_corpus():
    """Create a fiction corpus."""
    fiction_texts = [
        "The old wizard cast a powerful spell that illuminated the dark chamber.",
        "She gazed at the stars, wondering about the mysteries of the universe.",
        "The detective carefully examined the evidence at the crime scene.",
        "Magic flowed through the ancient runes carved into the stone walls.",
        "The spaceship navigated through the asteroid field with precision.",
        "Time seemed to slow as the hero faced the final challenge.",
        "The forest whispered secrets to those who knew how to listen.",
        "Destiny called to the young warrior standing at the crossroads.",
        "The ancient prophecy foretold the coming of the chosen one.",
        "Courage burned brightly in the hearts of the brave adventurers."
    ]
    
    corpus_file = "data/corpora/fiction_corpus.txt"
    Path(corpus_file).write_text("\n".join(fiction_texts), encoding="utf-8")
    print(f"✓ Created {corpus_file} with {len(fiction_texts)} fiction texts")
    return corpus_file

def create_business_corpus():
    """Create a business corpus."""
    business_texts = [
        "The quarterly report indicates strong growth in international markets.",
        "Our team successfully completed the project ahead of schedule.",
        "The conference featured presentations from industry experts.",
        "Strategic planning sessions identified new market opportunities.",
        "Customer feedback revealed areas for product improvement.",
        "The merger created synergies between complementary business units.",
        "Financial analysis showed positive return on investment metrics.",
        "Supply chain optimization reduced operational costs significantly.",
        "The marketing campaign generated substantial brand awareness.",
        "Employee training programs enhanced organizational capabilities."
    ]
    
    corpus_file = "data/corpora/business_corpus.txt"
    Path(corpus_file).write_text("\n".join(business_texts), encoding="utf-8")
    print(f"✓ Created {corpus_file} with {len(business_texts)} business texts")
    return corpus_file

def main():
    """Create diverse corpora for testing."""
    print("📚 Creating Diverse Corpora for Primitive Testing")
    print("=" * 60)
    
    # Create corpora directory
    Path("data/corpora").mkdir(exist_ok=True)
    
    # Create different types of corpora
    corpora = [
        create_news_corpus(),
        create_academic_corpus(),
        create_technical_corpus(),
        create_fiction_corpus(),
        create_business_corpus()
    ]
    
    print(f"\n📊 Summary:")
    print(f"Successfully created {len(corpora)} diverse corpora:")
    for corpus in corpora:
        print(f"  - {corpus}")
    
    return corpora

if __name__ == "__main__":
    main()

