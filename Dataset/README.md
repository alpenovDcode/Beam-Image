## Установка

Для начала необходимо установить необходимые библиотеки Python. Вы можете сделать это с помощью pip:

```bash
pip install -r requirements.txt
```
 
## Подготовка данных

Набор данных Flickr8k загружается и распаковывается с помощью библиотеки `opendatasets`.

```python
import opendatasets as od

od.download("https://www.kaggle.com/datasets/kunalgupta2616/flickr-8k-images-with-captions")
```