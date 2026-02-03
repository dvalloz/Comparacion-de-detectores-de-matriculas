#***Detector de Matriculas con Visión Artificial***
Autores del trabajo: Pablo López Martínez, David Valls Lozano y Daniel Ventura González


En este proyecto hemos desarrollado una serie de modelos de detección de matrículas basados en diferentes métodos de procesado de imágenes.

# **Detector** **Manual**

Este método se basa en la conversión de la imagen a escala de grises, para posteriormente obtener los contornos de la imagen e intentar localizar polígonos que aproximadente coincidan con la forma de una matrícula.

Utiliza como paquetes principales OpenCV y imutils para la mayoría de las funciones del método, junto a otros paquetes básicos.




    
![png](notebook_files/output_4_0.png)
    



    
![png](notebook_files/output_4_1.png)
    



    
![png](notebook_files/output_4_2.png)
    



    
![png](notebook_files/output_4_3.png)
    



    
![png](notebook_files/output_4_4.png)
    



    
![png](notebook_files/output_4_5.png)
    



    
![png](notebook_files/output_4_6.png)
    



    
![png](notebook_files/output_4_7.png)
    



    
![png](notebook_files/output_4_8.png)
    



    
![png](notebook_files/output_4_9.png)
    


    
    


# **Detector con EasyOCR**




    .
      


    coche4 → Sin detección
    


    
![png](notebook_files/output_6_5.png)
    


    coche11 → ['MS66YOB']
    


    
![png](notebook_files/output_6_9.png)
    




    matricula3 → ['4046JBB', '1398HKL', 'SP1514']
    


    
![png](notebook_files/output_6_13.png)
    



    


    


 
    
    


    Proceso completo. Resultados guardados en: todas_matriculas.txt
    

La primera función importante se encarga de comprobar si los fragmentos detectados se parecen a una matrícula siguiendo una serie de condiciones o patrones en base a la variedad de formatos de matrícula que hay en el mundo:

    - Matrículas que combinen letras y números, contando con un mínimo de 2 de cada una y con una longitud máxima combinada de 10 caracteres.
    
    - Matrículas que contienen únicamente letras, con una extensión entre 3 y 7 caracteres.

Después de esta función, está definida la encargada de procesar las imágenes.
Esta extrae los bloques de texto junto con su caja delimitadora y un nivel de probabilidad de confianza respecto a los caracteres obtenidos.
Además, se limpia el texto ante posibles errores en los que se detecten caracteres especiales como paréntesis o guiones que alteran los resultados y baja el porcentaje de acierto. Por otra parte, descartamos los bloques con probabilidad muy baja (inferiores a 0.3 sobre 1) o vacíos de contenido. Por último, se separan matrículas completas de fragmentos.

Para las detecciones de matrículas completas, dibujamos un rectángulo verde y añadimos el texto detectado.
Para aquellos fragmentos que forman parte de una posible predicción, se combinan ordenados de izquierda a derecha si cumplen alguno de los criterios y se dibujan el rectángulo y el texto de color púrpura para diferenciar.

Una vez terminado el proceso, guardamos la imagen resultante en la carpeta de salida y se crea una lista con las matrículas y la almacenamos en un archivo de texto. También imprimimos los resultados de texto detectados por consola para cada imagen.


# **Visualización de resultados**







    
![png](notebook_files/output_42_0.png)
    



    
![png](notebook_files/output_43_0.png)
    



# **Inferencia con los modelos entrenados**



    
![png](notebook_files/output_46_1.png)
    



    
![png](notebook_files/output_46_2.png)
    



    
![png](notebook_files/output_46_3.png)
    



    
![png](notebook_files/output_46_4.png)
    



    
![png](notebook_files/output_46_5.png)
    



    
![png](notebook_files/output_46_6.png)
    



    
![png](notebook_files/output_46_7.png)
    



    
![png](notebook_files/output_46_8.png)
    








    

# **Lectura y visualización de las matriculas con EasyOCR**




    
![png](notebook_files/output_49_4.png)
    


    Texto detectado: KLO1CA2555
    
    Procesando imagen: Cars1.jpg
    


    
![png](notebook_files/output_49_6.png)
    


    Texto detectado: PGoHN112
    
    Procesando imagen: Cars101.jpg
    


    
![png](notebook_files/output_49_8.png)
    


    Texto detectado: HR26 BC 5514
    
    Procesando imagen: Cars102.jpg
    


    
![png](notebook_files/output_49_10.png)
    


    Texto detectado: 68.611*36
    
    Procesando imagen: Cars105.jpg
    


    
![png](notebook_files/output_49_12.png)
    


    
    Procesando imagen: Cars117.jpg
    


    
![png](notebook_files/output_49_14.png)
    


    
    Procesando imagen: Cars121.jpg
    


    
![png](notebook_files/output_49_16.png)
    


    
    Procesando imagen: Cars131.jpg
    


    
![png](notebook_files/output_49_18.png)
    


    Texto detectado: @cGraPHICS
    Texto detectado: AP
    Texto detectado: 39
    Texto detectado: BP
    Texto detectado: 585
  


# **Bibliografia**



*   **Dataset:** https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data
*   **Repositorio de ejemplo:** https://www.kaggle.com/code/semihberaterdoan/license-plate-recognition-with-yolov11m/notebook
*   **Video sobre YOLO:** https://www.youtube.com/watch?v=ntoRvLgejUY
*   **Tutorial YOLO:** https://www.youtube.com/watch?v=LivJ-lzM-bM






