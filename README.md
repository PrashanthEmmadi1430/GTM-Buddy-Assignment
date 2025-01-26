## **Sales Call Classifier and Entity Extractor**

This project implements a multi-label classification model for sales call snippets and extracts domain-specific entities using dictionary lookup and SpaCy's Named Entity Recognition (NER). It includes a REST API, containerized with Docker, for inference.

### **Features**
- Synthetic dataset generation for sales call snippets.
- Multi-label classification for labels like:
  - Objection
  - Pricing Discussion
  - Security
  - Competition
- Entity extraction using:
  - Dictionary-based lookup from `domain_knowledge.json`.
  - NER with SpaCy.
- REST API for classification and entity extraction.
- Dockerized deployment.

---

### **Setup Instructions**

#### **1. Clone the Repository**
```bash
git clone https://github.com/PrashanthEmmadi1430/GTM-Buddy-Assignment
cd GTM-Buddy-Assignment
```

#### **2. Install Dependencies**
- Install Python dependencies using `pip`:
  ```bash
  pip install -r requirements.txt
  ```

- Download the SpaCy model:
  ```bash
  python -m spacy download en_core_web_sm
  ```

#### **3. Run Locally**
- Generate the synthetic dataset:
  ```bash
  python main.py
  ```
- The FastAPI server will start at `http://127.0.0.1:8000`.

#### **4. Test API**
Use the Swagger UI:
- Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Or use the following `curl` command:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text_snippet\": \"Can you provide a discount for your analytics product?\"}"
```

---

### **Docker Deployment**

#### **1. Build the Docker Image**
```bash
docker build -t salescall-api .
```

#### **2. Run the Docker Container**
```bash
docker run -p 8000:8000 salescall-api
```

#### **3. Test the API**
Once the container is running, use the following `curl` command to test:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text_snippet\": \"Can you provide a discount for your analytics product?\"}"
```

---

### **Project Files**
- **`main.py`**: Contains the core implementation and REST API.
- **`Dockerfile`**: Docker configuration for containerizing the application.
- **`requirements.txt`**: Python dependencies.
- **`domain_knowledge.json`**: Domain-specific keywords for entity extraction.
- **`calls_dataset.csv`**: Synthetic dataset for training the model.
- **`confusion_matrix.png`**: Visualizes model performance.


### **Future Improvements**
- Add more robust entity extraction using fine-tuned NER models.
- Extend domain knowledge with more categories and keywords.
- Deploy the containerized service to a cloud platform like AWS or Heroku.

