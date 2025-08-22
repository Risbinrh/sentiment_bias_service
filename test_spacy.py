#!/usr/bin/env python3
"""
Simple test script for spaCy entity recognition
"""

def test_spacy_ner():
    """Test spaCy entity recognition"""
    try:
        import spacy
        
        # Load model
        nlp = spacy.load("en_core_web_sm")
        print("✅ SpaCy model loaded successfully")
        
        # Test text
        text = "Apple Inc. announced that CEO Tim Cook will visit Singapore next month to meet with government officials."
        
        # Process text
        doc = nlp(text)
        
        # Extract entities
        entities = {
            "people": [],
            "organizations": [],
            "locations": []
        }
        
        for ent in doc.ents:
            print(f"Entity: '{ent.text}' - Label: {ent.label_}")
            
            if ent.label_ == "PERSON":
                entities["people"].append(ent.text)
            elif ent.label_ in ["ORG", "NORP"]:
                entities["organizations"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(ent.text)
        
        print(f"\n✅ Extracted entities: {entities}")
        return True
        
    except Exception as e:
        print(f"❌ SpaCy test failed: {e}")
        return False

if __name__ == "__main__":
    test_spacy_ner()