## 1. Title and Author
## Plant Disease Detection Using CNN.
- Prepared for: UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

- Author Name: Rashmini Akkapally

- GitHub Repository Link: [GitHub Repo](https://github.com/RashminiA6/UMBC-DATA606-Capstone)

- LinkedIn Profile Link: [LinkedIn Profile](https://www.linkedin.com/in/rashminiakkapally/)

- PowerPoint Presentation Link: [PowerPoint Presentation Link]

- YouTube Video Link: [YouTube Video Link]

---

## 2. Background

### What is it about?  
This project aims to predict plant diseases using images of plant leaves and provide solutions or precautions for the detected disease. The solution will be delivered through a user-friendly web application built with Streamlit. Farmers or users will be able to upload images of plant leaves, and the machine learning model will predict whether the plant is healthy or affected by a disease. The app will also suggest possible solutions to the problem if the plant is diseased.

### Why does it matter?  
Plant diseases pose a significant threat to global agriculture, causing major crop losses and impacting food security, especially in farming-dependent regions. Early detection is essential to prevent the spread of disease and ensure healthier crops. This project uses machine learning to create an accessible and efficient tool for diagnosing plant diseases, enabling farmers to take quick and effective action. It is particularly useful for beginners in farming, providing them with an easy-to-use solution for managing crops. By offering a scalable, cost-effective method, the project aims to improve agricultural productivity, reduce economic losses, and support sustainable food production.

### What are your research questions?  
1. Can machine learning effectively classify healthy and diseased plant leaves using image data?  
2. What level of accuracy can be achieved using deep learning techniques such as Convolutional Neural Networks (CNNs)?  
3. What are the most effective image preprocessing techniques for enhancing the accuracy of plant disease detection?

---

## 3. Data

### Data sources  
- The PlantVillage Dataset, available on Kaggle:  
  [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?select=segmented)  
  This dataset consists of over 50,000 expertly curated images of healthy and diseased leaves from various crop plants.

### Data size  
- Approximately 2.18 GB

### Data shape  
- Number of images: 54,305 images  
- Image dimensions: 256x256 pixels

### Time period (if applicable)  
- N/A (static image dataset without time-based observations)

### What does each row represent?  
Each row represents a single image of a crop plant leaf, which is either healthy or affected by a specific disease.

### Data dictionary  

| Column Name | Data Type   | Definition                                                | Potential Values                                 |
|-------------|-------------|------------------------------------------------------------|--------------------------------------------------|
| Image       | Image       | The plant leaf image                                       | N/A                                              |
| Label       | Categorical | The type of disease affecting the plant or 'Healthy'       | 'Healthy', 'Early Blight', 'Late Blight', 'Leaf Mold', etc. |

### Which variable/column will be your target/label in your ML model?  
- The `Label` column, which indicates whether the leaf is healthy or diseased and, if diseased, specifies the type of disease.

### Which variables/columns may be selected as features/predictors for your ML models?  
- The primary feature is the image data itself. This will be processed using image classification techniques, such as Convolutional Neural Networks (CNNs), to extract relevant features automatically.

