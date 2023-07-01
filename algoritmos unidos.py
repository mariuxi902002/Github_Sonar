import pandas as pd #importamos pandas
import sklearn #biblioteca de aprendizaje automático
import matplotlib.pyplot as plt #Librería especializada en la creación de gráficos
from sklearn.decomposition import PCA #importamos algorimo PCA
from sklearn.decomposition import IncrementalPCA #importamos algorimo PCA
from sklearn.decomposition import KernelPCA #importamos algorimo PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
from sklearn.impute import SimpleImputer#para valores faltantes

from sklearn.linear_model import LogisticRegression #clasificación y análisis predictivo 
from sklearn.preprocessing import StandardScaler #Normalizar los datos
from sklearn.preprocessing import KBinsDiscretizer#Discretizar los datos
from sklearn.model_selection import train_test_split #permite hacer una división de un conjunto de datos en dos 
import warnings

warnings.filterwarnings('ignore')
#bloques de entrenamiento y prueba de un modelo
if __name__ == '__main__':
    dt_heart=pd.read_csv('./data/dtf.csv')
    dt_heartnormalizados=pd.read_csv('./data/dtf.csv')
    dt_heartdiscretizados=pd.read_csv('./data/dtf.csv')
    
    #------------------ALGORITMOS PCA Y KERNELS-------------------

    #------------------SIN NORMALIZAR NI DISCRETIZAR-------------------
   #print(dt_heart.head(5)) #imprimimos los 5 primeros datos
    dt_features=dt_heart.drop(['%Toxicos'],axis=1) #las featurus sin el target
    dt_target = dt_heart['%Toxicos'] #obtenemos el target
   
    X_train,X_test,y_train,y_test =train_test_split(dt_features,dt_target,test_size=0.30,random_state=42)
    print(X_train.shape) #consultar la forma de la tabla con pandas
    print(y_train.shape)
    '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará 
    de esta forma:
    a: n_components = min(n_muestras, n_features)'''
    pca=PCA(n_components=3)
    # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    pca.fit(X_train)
    #Como haremos una comparación con incremental PCA, haremos 
    '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
    poco a poco y combinarlos en el resultado final'''
    ipca=IncrementalPCA(n_components=3,batch_size=10) #tamaño de bloques, no manda a entrear todos los datos
    #Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    ipca.fit(X_train)
    ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
    me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
    en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
    para nuestro modelo '''
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) #gneera desde 0 hasta los componentes
    #plt.show()
    #Ahora vamos a configurar nuestra regresión logística
    logistic=LogisticRegression(solver='lbfgs')
    # Configuramos los datos de entrenamiento
    dt_train = pca.transform(X_train)#conjunto de entrenamiento
    dt_test = pca.transform(X_test)#conjunto de prueba
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train) #mandasmos a regresion logistica los dos datasets
    #Calculamos nuestra exactitud de nuestra predicción
    print("========================PCA, IPCA, KPCA ============================")

    print("=======DATOS ORIGINALES=======")
    print("SCORE PCA: ", logistic.score(dt_test, y_test))
    #Configuramos los datos de entrenamiento
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    kernel = ['linear','poly','rbf']
    #Aplicamos la función de kernel de tipo polinomial
    for k in kernel:
        kpca = KernelPCA(n_components=4, kernel = k)
        #kpca = KernelPCA(n_components=4, kernel='poly' )
        #Vamos a ajustar los datos
        kpca.fit(X_train)
        #Aplicamos el algoritmo a nuestros datos de prueba y de entrenamiento
        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)
        #Aplicamos la regresión logística un vez que reducimos su dimensionalidad
        logistic = LogisticRegression(solver='lbfgs')
        #Entrenamos los datos
        logistic.fit(dt_train, y_train)
        #Imprimimos los resultados
        print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
    print("="*32)
    print("")



    #-----------------NORMALIZADO-------------------
    #print(dt_heart.head(5)) #imprimimos los 5 primeros datos
    dt_features=dt_heartnormalizados.drop(['%Toxicos'],axis=1) #las featurus sin el target
    dt_target = dt_heartnormalizados['%Toxicos'] #obtenemos el target

    dt_features = StandardScaler().fit_transform(dt_features) #Normalizamnos los datos

    X_train,X_test,y_train,y_test =train_test_split(dt_features,dt_target,test_size=0.30,random_state=42)
    pca=PCA(n_components=3)
    # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    pca.fit(X_train)
    ipca=IncrementalPCA(n_components=3,batch_size=10) #tamaño de bloques, no manda a entrear todos los datos
    ipca.fit(X_train)
    
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) #gneera desde 0 hasta los componentes
    #plt.show()
    #Ahora vamos a configurar nuestra regresión logística
    logistic=LogisticRegression(solver='lbfgs')
    # Configuramos los datos de entrenamiento
    dt_train = pca.transform(X_train)#conjunto de entrenamiento
    dt_test = pca.transform(X_test)#conjunto de prueba
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train) #mandasmos a regresion logistica los dos datasets
    #Calculamos nuestra exactitud de nuestra predicción
    print("=======DATOS NORMALIZADOS=======")
    print("SCORE PCA: ", logistic.score(dt_test, y_test))
    #Configuramos los datos de entrenamiento
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))
    kernel = ['linear','poly','rbf']
    #Aplicamos la función de kernel de tipo polinomial
    for k in kernel:
        kpca = KernelPCA(n_components=4, kernel = k)
        #kpca = KernelPCA(n_components=4, kernel='poly' )
        #Vamos a ajustar los datos
        kpca.fit(X_train)
        #Aplicamos el algoritmo a nuestros datos de prueba y de entrenamiento
        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)
        #Aplicamos la regresión logística un vez que reducimos su dimensionalidad
        logistic = LogisticRegression(solver='lbfgs')
        #Entrenamos los datos
        logistic.fit(dt_train, y_train)
        #Imprimimos los resultados
        print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
    print("="*32)
    print("")

    #------------------DISCRETIZADO-------------------
#print(dt_heart.head(5)) #imprimimos los 5 primeros datos
    dt_features = dt_heartdiscretizados.drop(['%Toxicos'], axis=1)
    dt_target = dt_heartdiscretizados['%Toxicos']
   # Crear objeto KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # Discretizar características
    dt_features_discretized = discretizer.fit_transform(dt_features)

    # Convertir la matriz discreta nuevamente en un DataFrame de pandas
    dt_features_discretized = pd.DataFrame(dt_features_discretized, columns=dt_features.columns)

    X_train, X_test, y_train, y_test = train_test_split(dt_features_discretized, dt_target, test_size=0.30, random_state=42)
    pca = PCA(n_components=3)
    pca.fit(X_train)
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    logistic = LogisticRegression(solver='lbfgs')
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("=======DATOS  DISCRETIZADOS=======")
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    kernel = ['linear', 'poly', 'rbf']
    for k in kernel:
        kpca = KernelPCA(n_components=4, kernel=k)
        kpca.fit(X_train)
        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)
        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(dt_train, y_train)
        print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
    print("="*32)


    print("")
    print("")
    print("========================REGULARIZACION============================")


 
    dataset = pd.read_csv('./data/dtf.csv')

    # Separar las características (X) y la variable objetivo (y)
    X = dataset[['Total_commits', 'Total_commits_per_day', 'Accumulated_commits', 'Committers', 'Committers_Weight', 'classes', 'Committers', 'functions', 'duplicated_lines', 'test_errors', 'skipped_tests', 'coverage', 'complexity', 'comment_lines', 'comment_lines_density', 'duplicated_lines_density', 'files', 'directories', 'file_complexity', 'violations', 'duplicated_blocks', 'duplicated_files', 'lines', 'public_api', 'statements', 'blocker_violations', 'critical_violations', 'major_violations', 'minor_violations', 'info_violations', 'lines_to_cover', 'line_coverage', 'conditions_to_cover', 'branch_coverage', 'sqale_index', 'sqale_rating', 'false_positive_issues', 'open_issues', 'reopened_issues', 'confirmed_issues', 'sqale_debt_ratio', 'new_sqale_debt_ratio', 'code_smells', 'new_code_smells', 'bugs', 'effort_to_reach_maintainability_rating_a', 'reliability_remediation_effort', 'reliability_rating', 'security_remediation_effort', 'security_rating', 'cognitive_complexity', 'new_development_cost', 'security_hotspots', 'security_review_rating']]
    y = dataset[['%Toxicos']]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Discretizar los datos
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    X_train_discretized = discretizer.fit_transform(X_train)
    X_test_discretized = discretizer.transform(X_test)

    # Entrenar los modelos con datos originales
    modelLinear = LinearRegression().fit(X_train, y_train)
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    modelElasticNet = ElasticNet(random_state=0).fit(X_train, y_train)

    # Hacer predicciones con datos originales
    y_predict_linear = modelLinear.predict(X_test)
    y_predict_lasso = modelLasso.predict(X_test)
    y_predict_ridge = modelRidge.predict(X_test)
    y_pred_elastic = modelElasticNet.predict(X_test)

    # Calcular las pérdidas con datos originales
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)

    # Imprimir las pérdidas con datos originales
    print("=======PERDIDAS CON DATOS ORIGINALES=======")
    print("Linear Loss:", linear_loss)
    print("Lasso Loss:", lasso_loss)
    print("Ridge Loss:", ridge_loss)
    print("ElasticNet:", elastic_loss)
    print("="*32)
    print("")


    # Entrenar los modelos con datos normalizados
    modelLinear_normalized = LinearRegression().fit(X_train_normalized, y_train)
    modelLasso_normalized = Lasso(alpha=0.2).fit(X_train_normalized, y_train)
    modelRidge_normalized = Ridge(alpha=1).fit(X_train_normalized, y_train)
    modelElasticNet_normalized = ElasticNet(random_state=0).fit(X_train_normalized, y_train)

    # Hacer predicciones con datos normalizados
    y_predict_linear_normalized = modelLinear_normalized.predict(X_test_normalized)
    y_predict_lasso_normalized = modelLasso_normalized.predict(X_test_normalized)
    y_predict_ridge_normalized = modelRidge_normalized.predict(X_test_normalized)
    y_pred_elastic_normalized = modelElasticNet_normalized.predict(X_test_normalized)

    # Calcular las pérdidas con datos normalizados
    linear_loss_normalized = mean_squared_error(y_test, y_predict_linear_normalized)
    lasso_loss_normalized = mean_squared_error(y_test, y_predict_lasso_normalized)
    ridge_loss_normalized = mean_squared_error(y_test, y_predict_ridge_normalized)
    elastic_loss_normalized = mean_squared_error(y_test, y_pred_elastic_normalized)

    # Imprimir las pérdidas con datos normalizados
    print("======PERDIDAS CON DATOS NORMALIZADOS=======")
    print("Linear Loss:", linear_loss_normalized)
    print("Lasso Loss:", lasso_loss_normalized)
    print("Ridge Loss:", ridge_loss_normalized)
    print("ElasticNet:", elastic_loss_normalized)
    print("="*32)
    print("")


    # Entrenar los modelos con datos discretizados
    modelLinear_discretized = LinearRegression().fit(X_train_discretized, y_train)
    modelLasso_discretized = Lasso(alpha=0.2).fit(X_train_discretized, y_train)
    modelRidge_discretized = Ridge(alpha=1).fit(X_train_discretized, y_train)
    modelElasticNet_discretized = ElasticNet(random_state=0).fit(X_train_discretized, y_train)

    # Hacer predicciones con datos discretizados
    y_predict_linear_discretized = modelLinear_discretized.predict(X_test_discretized)
    y_predict_lasso_discretized = modelLasso_discretized.predict(X_test_discretized)
    y_predict_ridge_discretized = modelRidge_discretized.predict(X_test_discretized)
    y_pred_elastic_discretized = modelElasticNet_discretized.predict(X_test_discretized)

    # Calcular las pérdidas con datos discretizados
    linear_loss_discretized = mean_squared_error(y_test, y_predict_linear_discretized)
    lasso_loss_discretized = mean_squared_error(y_test, y_predict_lasso_discretized)
    ridge_loss_discretized = mean_squared_error(y_test, y_predict_ridge_discretized)
    elastic_loss_discretized = mean_squared_error(y_test, y_pred_elastic_discretized)



    # Imprimir las pérdidas con datos discretizados
    print("=======PERDIDAS CON DATOS DISCRETIZADOS=======")
    print("Linear Loss", linear_loss_discretized)
    print("Lasso Loss):", lasso_loss_discretized)
    print("Ridge Loss:", ridge_loss_discretized)
    print("ElasticNet:", elastic_loss_discretized)
    print("="*32)
    print("")

    print("=======COEFICIENTES CON DATOS ORIGINALES=======")
     # Imprimir los coeficientes de los modelos con datos originales
    print("Coeficientes Linear:", modelLinear.coef_)
    print("Coeficientes Lasso:", modelLasso.coef_)
    print("Coeficientes Ridge:", modelRidge.coef_)
    print("Coeficientes ElasticNet:", modelElasticNet.coef_)
    print("=" * 128)

    print("=======COEFICIENTES CON DATOS NORMALIZADOS=======")
    # Imprimir los coeficientes de los modelos con datos normalizados
    print("Coeficientes Linear:", modelLinear_normalized.coef_)
    print("Coeficientes Lasso:", modelLasso_normalized.coef_)
    print("Coeficientes Ridge:", modelRidge_normalized.coef_)
    print("Coeficientes ElasticNet:", modelElasticNet_normalized.coef_)
    print("=" * 128)


    print("=======COEFICIENTES CON DATOS DISCRETIZADOS=======")
    # Imprimir los coeficientes de los modelos con datos discretizados
    print("Coeficientes Linear:", modelLinear_discretized.coef_)
    print("Coeficientes Lasso:", modelLasso_discretized.coef_)
    print("Coeficientes Ridge", modelRidge_discretized.coef_)
    print("Coeficientes ElasticNet:", modelElasticNet_discretized.coef_)
    print("=" * 128)





    # Imprimir los puntajes (scores) de los modelos en los conjuntos de datos originales
    print("=======SCORE CON DATOS ORIGINALES=======")
    print("Score Linear:", modelLinear.score(X_test, y_test))
    print("Score Lasso:", modelLasso.score(X_test, y_test))
    print("Score Ridge:", modelRidge.score(X_test, y_test))
    print("Score ElasticNet:", modelElasticNet.score(X_test, y_test))
    print("="*32)
    print("")


    # Imprimir los puntajes (scores) de los modelos en los conjuntos de datos normalizados
    print("=======SCORE CON DATOS NORMALIZADOS=======")
    print("Score Linear:", modelLinear_normalized.score(X_test_normalized, y_test))
    print("Score Lasso:", modelLasso_normalized.score(X_test_normalized, y_test))
    print("Score Ridge:", modelRidge_normalized.score(X_test_normalized, y_test))
    print("Score ElasticNet:", modelElasticNet_normalized.score(X_test_normalized, y_test))
    print("="*32)
    print("")

    # Imprimir los puntajes (scores) de los modelos en los conjuntos de datos discretizados
    print("=======SCORE CON DATOS DISCRETIZADOS=======")
    print("Score Linear:", modelLinear_discretized.score(X_test_discretized, y_test))
    print("Score Lasso:", modelLasso_discretized.score(X_test_discretized, y_test))
    print("Score Ridge:", modelRidge_discretized.score(X_test_discretized, y_test))
    print("Score ElasticNet:", modelElasticNet_discretized.score(X_test_discretized, y_test))
    print("="*32)
    print("")
    print("")
    print("========================VALORES ATIPICOS============================")

    np.random.seed(42)  # Fijar la semilla aleatoria para reproducibilidad

    dataset = pd.read_csv('./data/dtf.csv')
    #print(dataset.head(5))

    # Datos originales
    X_orig = dataset.drop(['security_review_rating', '%Toxicos'], axis=1)
    y_orig = dataset[['%Toxicos']]

    # Normalización de los datos
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_orig)
    y_normalized = scaler.fit_transform(y_orig)

    # Discretización de los datos
    X_discretized = pd.DataFrame()
    for column in X_orig.columns:
        discretized_col = pd.cut(X_orig[column], bins=5, labels=False)
        X_discretized[column] = discretized_col
    y_discretized = pd.cut(y_orig.squeeze(), bins=5, labels=False)

    # División de los datos en conjuntos de entrenamiento y prueba
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.3, random_state=42
    )
    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(
        X_normalized, y_normalized, test_size=0.3, random_state=42
    )
    X_train_discretized, X_test_discretized, y_train_discretized, y_test_discretized = train_test_split(
        X_discretized, y_discretized, test_size=0.3, random_state=42
    )

    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    warnings.simplefilter("ignore")

    for name, estimator in estimators.items():
        print("=============",name,"=============")

        # Entrenamiento con datos originales
        estimator.fit(X_train_orig, y_train_orig)
        predictions_orig = estimator.predict(X_test_orig)
        mse_orig = mean_squared_error(y_test_orig, predictions_orig)
        print("ORIGINALES: ", mse_orig)

        plt.ylabel('Puntaje Predicho')
        plt.xlabel('Puntaje Real')
        plt.title('Predicho vs Real (Originales)')
        plt.scatter(y_test_orig, predictions_orig)
        plt.plot(predictions_orig, predictions_orig, 'r--')
        #plt.show()
    

        # Entrenamiento con datos normalizados
        estimator.fit(X_train_normalized, y_train_normalized)
        predictions_normalized = estimator.predict(X_test_normalized)
        mse_normalized = mean_squared_error(y_test_normalized, predictions_normalized)
        print("NORMALIZADO: ", mse_normalized)

        plt.ylabel('Puntaje Predicho')
        plt.xlabel('Puntaje Real')
        plt.title('Predicho vs Real (Normalizado)')
        plt.scatter(y_test_normalized, predictions_normalized)
        plt.plot(predictions_normalized, predictions_normalized, 'r--')
        #plt.show()

        # Entrenamiento con datos discretizados
        estimator.fit(X_train_discretized, y_train_discretized)
        predictions_discretized = estimator.predict(X_test_discretized)
        mse_discretized = mean_squared_error(y_test_discretized, predictions_discretized)
        print("DISCRETIZADO: ", mse_discretized)

        plt.ylabel('Puntaje Predicho')
        plt.xlabel('Puntaje Real')
        plt.title('Predicho vs Real (Discretizado)')
        plt.scatter(y_test_discretized, predictions_discretized)
        plt.plot(predictions_discretized, predictions_discretized, 'r--')
        #plt.show()
        print("="*32)
        print("")



    print("")
    print("========================METODOS DE ENSAMBLE============================")
    print("========================BAGGING============================")
    np.random.seed(42)  # Semilla para reproducibilidad

    dt_heart = pd.read_csv('./data/dtf.csv')
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']

    # Datos originales
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)
    
    # Normalización de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    
    # Discretización de datos
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_orig)
    X_test_discretized = discretizer.transform(X_test_orig)

    estimators = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN': KNeighborsClassifier(),
        'DecisionTreeClf': DecisionTreeClassifier(),
        'RandomTreeForest': RandomForestClassifier(random_state=0)
    }


    print('=======SCORE ORIGINALES============')
    
    for name, estimator in estimators.items():
        estimator.fit(X_train_orig, y_train)
        y_pred = estimator.predict(X_test_orig)
        print('SCORE {} : {}'.format(name, accuracy_score(y_pred, y_test)))

    print('=' * 32)
    print("")
    print('=======SCORE NORMALIZADOS============')
    
    for name, estimator in estimators.items():
        estimator.fit(X_train_scaled, y_train)
        y_pred = estimator.predict(X_test_scaled)
        print('SCORE {} : {}'.format(name, accuracy_score(y_pred, y_test)))

    print('=' * 32)
    print("")
    print('=======SCORE DISCRETOS============')
    
    for name, estimator in estimators.items():
        estimator.fit(X_train_discretized, y_train)
        y_pred = estimator.predict(X_test_discretized)
        print('SCORE {} : {}'.format(name, accuracy_score(y_pred, y_test)))
    print('=' * 32)
    print("")

    print("========================BOOSTING============================")
    dt_heart = pd.read_csv('./data/dtf.csv')
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_scaled)
    X_test_discretized = discretizer.transform(X_test_scaled)

    estimators = range(2, 300, 2)
    total_accuracy = []
    best_result_original = {'result' : 0, 'n_estimator': 1}
    best_result_normalized = {'result' : 0, 'n_estimator': 1}
    best_result_discretized = {'result' : 0, 'n_estimator': 1}
    for i in estimators:
        # Ajustar el modelo a los datos originales
        boost_original = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred_original = boost_original.predict(X_test)
        accuracy_original = accuracy_score(boost_pred_original, y_test)
        if accuracy_original > best_result_original['result']:
            best_result_original['result'] = accuracy_original
            best_result_original['n_estimator'] = i

        # Ajustar el modelo a los datos normalizados
        boost_normalized = GradientBoostingClassifier(n_estimators=i).fit(X_train_scaled, y_train)
        boost_pred_normalized = boost_normalized.predict(X_test_scaled)
        accuracy_normalized = accuracy_score(boost_pred_normalized, y_test)
        if accuracy_normalized > best_result_normalized['result']:
            best_result_normalized['result'] = accuracy_normalized
            best_result_normalized['n_estimator'] = i

        # Ajustar el modelo a los datos discretizados
        boost_discretized = GradientBoostingClassifier(n_estimators=i).fit(X_train_discretized, y_train)
        boost_pred_discretized = boost_discretized.predict(X_test_discretized)
        accuracy_discretized = accuracy_score(boost_pred_discretized, y_test)
        if accuracy_discretized > best_result_discretized['result']:
            best_result_discretized['result'] = accuracy_discretized
            best_result_discretized['n_estimator'] = i

    # Imprimir los mejores resultados para cada conjunto de datos
    print(f'Mejor resultado con datos originales: {best_result_original}')
    print(f'Mejor resultado con datos originales: {best_result_normalized}')
    print(f'Mejor resultado con datos originales: {best_result_discretized}')

