# 🖼️ Image Retrieval


## 📄 Summary

This project demonstrates how to classify and retrieve similar images using a neural network on the Caltech101 dataset. The goal is to achieve **top-k accuracy (k=5)** greater than **90%**, which was accomplished by fine-tuning a pretrained **ResNet-50** model.


## 📊 Dataset Info

The [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) dataset consists of 101 object categories plus one background clutter category. There are roughly **40 to 800 images per class**, including examples such as:

- Faces
- Airplanes
- Beavers
- Mandolins  
…and many more.

## Evaluation + Conclusion:
* Achieved a top-5 accuracy of 92.4% on the validation set using a fine-tuned ResNet-50.
* Feature extraction and retrieval using FAISS was efficient and accurate, with most top-5 retrievals belonging to the correct class.
* Streamlit app demonstrates real-time image similarity search using precomputed features.
* Potential future improvements:
    * Use a more compact model for faster inference.
    * Add support for user-uploaded images.


## 🚀 Streamlit App
### 🔗 Live Demo

Try the app here: [Image Retrieval Streamlit App]()

![Demo Screenshot](<img width="1474" alt="web-app" src="https://github.com/user-attachments/assets/e2848363-3822-49b7-a137-6bf5e4c97a75" />
)


To run the interactive image retrieval demo:

```bash
Before running the app, make sure Docker is up and running (if you’re using Docker for your environment). Then, run the following to precompute features:
./utils/precompute.sh

After precomputing, start the app with:
streamlit run app.py
If you see this error:

OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

✅ Temporary Fix
Before running the app, run this in your terminal:
export KMP_DUPLICATE_LIB_OK=TRUE

Or run the app inline like this:

KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py --server.fileWatcherType none

