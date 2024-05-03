# Shakkelly - شَكِّلْ لِي
Shakkelly (شَكِّلْ لِي) is a project aims to restore Arabic text diacritization (تشكيل) using deep learning. Diacritizing Arabic text has a lot of applications like help text-to-speach accuracy, improving search results and help individuals fastly diacritize their writings.

## Dataset info
[Tashkeela Clean](https://www.kaggle.com/datasets/ahmedmohsen2002/tashkeela-clean-arabic-diacritized-corpus): Is a clean version of [ Tashkeela: Novel corpus of Arabic vocalized texts, data for auto-diacritization systems](https://www.sciencedirect.com/science/article/pii/S2352340917300112]) which contains data with over 75 million of fully vocalized words obtained from 97 books, structured in text files. the data has been cleaned with several methods and over multiple version that is detailed in a changelog file attached with the dataset documenting all the specific changes made over all version.

## Model Info
- The currently implemented model uses bidirectional RNN layers (LSTM or GRU).
- In the future, more models architecitures will be used such as Attention based models to achieve best results.

## Project Setup
1- Clone this repository:
```bash
git clone https://github.com/PrinceEGY/Shakkelly.git
cd Shakkelly
```
2- Set up environment:
```bash
pip install -r requirements.txt
```
## Usage
- Using Python environment
```python
from modules import Diacritizer

diacritizer = Diacritizer()
print(diacritizer("السلام عليكم ورحمة الله"))
# السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ
```

- API endpoint: "https://shakkelly.onrender.com/shakkel"
```python
import requests
result = requests.post(
    "https://shakkelly.onrender.com/shakkel",
    json={"text": "السلام عليكم ورحمة الله"},
)
print(result)
# {'diacritized': 'السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ'}
```
## Some examples
| Real diacritization | Predicted diacritization |
|---------------------|--------------------------|
|وَإِنْ قُلْنَا يَخْرُجُونَ مِنْ الْمَسْجِدِ وَلَا يَجْمَعُونَ مَعَهُمْ فَرُبَّمَا لَا يَتَيَسَّرُ لَهُمْ صَلَاتُهَا جَمَاعَةً | وَإِنْ قُلْنَا يَخْرُجُونَ مِنْ الْمَسْجِدِ وَلَا يَجْمَعُونَ مَعَهُمْ فَرُبَّمَا لَا يَتَيَسَّرُ لَهُمْ صَلَاتُهَا جَمَاعَةٌ
|بَرَزَ الثَّعْلَبُ يَوْمًا فِي شِعَارِ الْوَاعِظِينَا | بَرَزَ الثَّعْلَبُ يَوْمًا فِي شِعَارِ الْوَاعِظِينَا
|لِأَنَّهُ أَقَرَّ بِشَيْئَيْنِ مُبْهَمَيْنِ وَعَقَّبَهُمَا بِالدِّرْهَمِ مَنْصُوبًا فَالظَّاهِرُ أَنَّهُ تَفْسِيرٌ لِكُلٍّ مِنْهُمَا |لِأَنَّهُ أَقَرَّ بِشَيْئَيْنِ مُبْهِمَيْنِ وَعَقِبَهُمَا بِالدِّرْهَمِ مَنْصُوبًا فَالظَّاهِرُ أَنَّهُ تَفْسِيرٌ لِكُلٍّ مِنْهُمَا
