# Lawn paths recognition system


## Installation

Install from repository:
~~~
pip install -e "git+https://github.com/Denikozub/lawn_paths.git#egg=lawn_paths"
~~~
Warning: package requires GeoPandas to be installed, which can be problematic on Windows. [This](https://towardsdatascience.com/geopandas-installation-the-easy-way-for-windows-31a666b3610f/) article may help.


## Docker

~~~
docker run -d -it —name final —mount type=bind,source="$(pwd)"/target,target=/app kadmus_map
~~~
target/ - директория с файлами TIF и TFW  
[Ссылка на Docker](https://disk.yandex.ru/d/32I0DB2AxfSvDw)  


## Description

### Shapefile
[Documentation](https://github.com/Denikozub/kadmus-dev#build-shapefile)  
Программа строит шейпфайл по результатам работы нейросети - NPY маскам изображений и соответствующим TFW файлам мира.
Имеется возможность фильтровать полученные результаты, сглаживать итоговые линии и выбирать систему координат.  

Фильтрация:
* По ширине
* По расстоянию между тропами
* По размерам bounding box
* По площади  

Дополнительно имеется возможность эффективно (6 секунд на тестовых данных) проверять пересечения со зданиями Москвы для уменьшения вероятности ошибки. Данные скачаны из открытого источника [OpenStreetMap](www.openstreetmap.org), обработаны и сжаты.

### Visualization
[Documentation](https://github.com/Denikozub/kadmus-dev#visualize)  
Реализована возможность нанесение найденных троп на интерактивную карту OpenStreetMap с помощью сервиса с открытым исходным кодом [Leaflet](https://leafletjs.com/). Данное решение не требует платного API таких сервисов, как Google Maps или Yandex Maps, и может быть использовано в коммерческих проектах.

### Iterative Makrup
[Documentation](https://github.com/Denikozub/kadmus-dev#preliminary-markup)  
Разработан и реализован итеративный подход к разметке данных с применением глубокого обучения:  
1. Разметить часть данных вручную
2. Обучить нейросеть на размеченных данных
3. Применить нейросеть для помощи при разметке следующей части данных
4. Перейти на шаг 1  


## Documentation

### Application

~~~
python visual_build/main.py
~~~

### Preliminary Markup

Запуск из командной строки:  
~~~
python preliminary_markup/pipeline.py get_mask img.tif
~~~
Возвращает маску, где красным цветом на белом фоне выделены области, которые нашла нейросеть.  
Маска сохраняется в img_mask.tif  

Остается стереть лишние красные метки, затем сконвертировать маску в .npy; команда для конвертации:
~~~
python preliminary_markup/pipeline.py get_npy img_mask.tif
~~~
Красный цвет заменяет на белый, все остальное - на черный.  

Команда чтобы посмотреть результат наложения маски и изображения:
~~~
python preliminary_markup/pipeline.py blend image.tif mask.tif  
~~~
Для каждого следующего этапа обучения нейросети необходимо обновлять pipeline.


### Build shapefile

~~~python
from lawn_paths.map_builder.shapefile import build_shapefile

build_shapefile(dataset_directory: str,
                file_list: list = None,
                buildings_file: str = "Moscow_Buildings.zip",
                output_filename: str = 'paths.shp',
                crs: str = 'epsg:32637',
                max_path_distance_cm: float = 100.,
                max_path_width_cm: float = 80.,
                min_bbox_size_m: float = 1.,
                max_bbox_size_m: float = 200.,
                max_path_area_m2: float = 100.,
                p_epsilon: float = 0.3,
                c_epsilon: float = 2.)
~~~

__dataset_directory__: directory where .NPY mask files and .TFW world files are contained  
__file_list__: list of filenames to be processed (without extensions)  
__buildings_file__ file with buildings (polygons) in epsg:4326  
__output_filename__: name of the output file (should be .SHP)  
__crs__: initial coordinate reference system  
__max_path_distance_cm__: max distance between paths for them to be connected in cm  
__max_path_width_cm__: max path width  
__min_bbox_size_m__: min size of path's bounding box in meters  
__max_bbox_size_m__: maxsize of path's bounding box in meters  
__max_path_area_m2__ max path area in squared meters  
__p_epsilon__: RDP parameter to smooth path polygons  
__c_epsilon__: RDP parameter to smooth path polygon centerlines  


### Visualize

~~~python
from lawn_paths.map_builder.visualize import visualize

visualize(filename: str, output_file: str)
~~~
matplotlib==3.3.2 required  
__filename__ path to SHP file with paths to visualize  
__output_file__ path to HTML file with interactive map  
