from main import Classification
from datetime import datetime as time
import os
import time as t
import random as rnd


# --------------------------------------------------------------------------------------------
def Print_CSV(ti):
    print("\n\n\n\n\n\n\n\n\n")
    print("\t\t\t\t\t     _______       _______                       ")
    print("\t\t\t\t\t    /       \     /       \   \              /   ")
    print("\t\t\t\t\t   /         \   |             \            /    ")
    print("\t\t\t\t\t  /              |              \          /     ")
    print("\t\t\t\t\t |                \_______       \        /      ")
    print("\t\t\t\t\t |                        \       \      /       ")
    print("\t\t\t\t\t  \                        |       \    /        ")
    print("\t\t\t\t\t   \        /              |        \  /         ")
    print("\t\t\t\t\t    \______/      \_______/          \/          ")
    print("\t\t\t\t\t")
    t.sleep(ti)
    os.system('cls')


# --------------------------------------------------------------------------------------------
def Print_SVM(ti):
    print("\n\n\n\n\n\n\n\n\n")
    print("\t\t\t\t\t  _______")
    print("\t\t\t\t\t /       \   \              /   |\              /| ")
    print("\t\t\t\t\t|             \            /    | \            / | ")
    print("\t\t\t\t\t|              \          /     |  \          /  | ")
    print("\t\t\t\t\t \_______       \        /      |   \        /   | ")
    print("\t\t\t\t\t         \       \      /       |    \      /    | ")
    print("\t\t\t\t\t          |       \    /        |     \    /     | ")
    print("\t\t\t\t\t          |        \  /         |      \  /      | ")
    print("\t\t\t\t\t \_______/          \/          |       \/       | ")
    print("\t\t\t\t\t")
    t.sleep(ti)
    os.system('cls')


# --------------------------------------------------------------------------------------------
def Print_NB(ti):
    print("\n\n\n\n\n\n\n\n\n")
    print("\t\t\t\t\t\t              ______       ")
    print("\t\t\t\t\t\t|\       |   |      \      ")
    print("\t\t\t\t\t\t| \      |   |       |     ")
    print("\t\t\t\t\t\t|  \     |   |       |     ")
    print("\t\t\t\t\t\t|   \    |   |_____ /      ")
    print("\t\t\t\t\t\t|    \   |   |       \     ")
    print("\t\t\t\t\t\t|     \  |   |        |    ")
    print("\t\t\t\t\t\t|      \ |   |        |    ")
    print("\t\t\t\t\t\t|       \|.  |_______/     ")
    t.sleep(ti)
    os.system('cls')


# --------------------------------------------------------------------------------------------
def Print_KNN(ti):
    print("\n\n\n\n\n\n\n\n\n")
    print("\t\t\t\t\t\t|    /\   |\       |   |\       |    ")
    print("\t\t\t\t\t\t|   /     | \      |   | \      |    ")
    print("\t\t\t\t\t\t|  /      |  \     |   |  \     |    ")
    print("\t\t\t\t\t\t| /       |   \    |   |   \    |    ")
    print("\t\t\t\t\t\t| \       |    \   |   |    \   |    ")
    print("\t\t\t\t\t\t|  \      |     \  |   |     \  |    ")
    print("\t\t\t\t\t\t|   \     |      \ |   |      \ |    ")
    print("\t\t\t\t\t\t|    \__  |       \|   |       \|    ")
    print("\t\t\t")
    t.sleep(ti)
    os.system('cls')


# --------------------------------------------------------------------------------------------------------------
def Print_DT(ti):
    print("\n\n\n\n\n\n\n\n\n")
    print("\t\t\t\t\t\t_______      _____________     ")
    print("\t\t\t\t\t\t|      \     |     |     |     ")
    print("\t\t\t\t\t\t|       \          |           ")
    print("\t\t\t\t\t\t|        \         |           ")
    print("\t\t\t\t\t\t|         |        |           ")
    print("\t\t\t\t\t\t|         |        |           ")
    print("\t\t\t\t\t\t|        /         |           ")
    print("\t\t\t\t\t\t|       /          |           ")
    print("\t\t\t\t\t\t|______/    .      |__         ")
    t.sleep(ti)
    os.system('cls')


# -------------------------------------------------------------------------------------------------------------------------------------
def Print_KMEANS(ti):
    print("\n\n\n\n\n\n\n\n\n")
    print("\t\t\t                              __________                                   ________      ")
    print("\t\t\t|    /\  |\              /|   |        |         /\         |\       |    /        \     ")
    print("\t\t\t|   /    | \            / |   |                 /  \        | \      |   |               ")
    print("\t\t\t|  /     |  \          /  |   |                /    \       |  \     |   |               ")
    print("\t\t\t| /      |   \        /   |   |_________      /      \      |   \    |    \________      ")
    print("\t\t\t| \      |    \      /    |   |              /________\     |    \   |             \     ")
    print("\t\t\t|  \     |     \    /     |   |             /          \    |     \  |              |    ")
    print("\t\t\t|   \    |      \  /      |   |            /            \   |      \ |              |    ")
    print("\t\t\t|    \__ |       \/       |   |________|  /              \  |       \|    \________/     ")
    print("\t\t\t\t")
    t.sleep(ti)
    os.system('cls')


# -------------------------------------------------------------------------------------------------------------------------------
def Print_Menu():
    print("\n\n\n\t\t\t\t   \t  -------- Classifications Menu --------   ")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t1. Build Data \t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t2. SVM \t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t3. Naive Bayes \t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t4. KNN \t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t5. Decision Tree \t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t6. Exit \t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t ---------------------------------------- ")


# --------------------------------------------------------------------------------------------
def Print_Main_Menu():
    print("\n\n\n\t\t\t\t   \t  -------- Data Mining Project ---------  ")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t1. Classifications Menu\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t2. Train All Classifiers\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t3. Training Tables\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t4. Repeat Training\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t5. Find Best Parameters\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t6. Kmeans Clustring\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t7. Classifiers LOGO\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t8. Exit\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t ---------------------------------------- ")


# --------------------------------------------------------------------------------------------
def Print_Table_Menu():
    print("\n\n\n\t\t\t\t   \t  ----------- Training Tables ----------   ")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t1. SVM \t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t2. Naive Bayes \t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t3. KNN \t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t4. Decision Tree \t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t5. Exit \t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t | \t\t\t\t\t|")
    print("\t\t\t\t\t ---------------------------------------- ")


# --------------------------------------------------------------------------------------------------------------------------------------------
prt_tbl_delay = 0.1


def Print_SVM_Table(r, g, c, tre, tse, ti, tmp):
    t.sleep(prt_tbl_delay)
    disp = ''
    star = ''
    diff = round(tre - tse, 5)
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    if diff == tmp:
        disp = ' BEST'
    g = str(g)
    c = str(c)
    tre = str(tre)
    tse = str(tse)
    ti = str(ti)
    repe = str(r)
    if r == 1:
        print("")
        print("\t\t          ____________          _-------------------------------------------------------------_")
        print("\t\t         |    SVM     |        |                   Performance Training Table                  |")
        print(
            "\t\t ________|____________|________|_______________________________________________________________|________")
        print(
            "\t\t|   ##   |   Kernel   |   Parameters     |  Train Error  |   Test Error  |  Train Time   |    Status    |")
        print(
            "\t\t|________|____________|__________________|_______________|_______________|_______________|______________|")
    print(
        "\t\t|   " + repe + "\t |    RBF     | G=" + g + ",C=" + c + "\t |    " + tre + " %\t |    " + tse + " %\t |    " + ti + " ms\t |     " + star + "\t|" + disp + " " + str(
            diff))
    print(
        "\t\t|________|____________|__________________|_______________|_______________|_______________|______________|")


# ----------------------------------------------------------------------------------------------------------------------------------------------
def Print_NB_Table(r, tre, tse, ti, tmp):
    t.sleep(prt_tbl_delay)
    disp = ''
    diff = round(tre - tse, 5)
    star = ''
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    if diff == tmp:
        disp = ' BEST'
    tre = str(tre)
    tse = str(tse)
    ti = str(ti)
    repe = str(r)
    if r == 1:
        print("\n\n\n")
        print("\t\t        _----------------------------------------------------------------------_")
        print("\t\t       |                        Performance Training Table                      |")
        print("\t\t ______|________________________________________________________________________|_______")
        print("\t\t|   ##   |      N.B      |  Train Error  |   Test Error  |  Train Time   |    Status    |")
        print("\t\t|________|_______________|_______________|_______________|_______________|______________|")
    print(
        "\t\t|   " + repe + "\t | Train Perform |    " + tre + " %\t |    " + tse + " %\t |    " + ti + " ms\t |     " + star + "\t|" + disp + " " + str(
            diff))
    print("\t\t|________|_______________|_______________|_______________|_______________|______________|")


# ---------------------------------------------------------------------------------------------------------------------------------------
def Print_KNN_Table(r, k, tre, tse, ti, tmp):
    t.sleep(prt_tbl_delay)
    disp = ''
    star = ''
    diff = round(tre - tse, 5)
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    if diff == tmp:
        disp = ' BEST'
    k = str(k)
    tre = str(tre)
    tse = str(tse)
    ti = str(ti)
    repe = str(r)
    if r == 1:
        print("\n\n\n")
        print("\t\t        _------------------------------------------------------------------------------_")
        print("\t\t       |                            Performance Training Table                          |")
        print("\t\t ______|________________________________________________________________________________|_______")
        print("\t\t|   ##   |    K.N.N    |Parameter|  Train Error  |   Test Error  |  Train Time   |    Status    |")
        print("\t\t|________|_____________|_________|_______________|_______________|_______________|______________|")
    print(
        "\t\t|   " + repe + "\t | Performance | Ki=" + k + "\t |    " + tre + " %\t |    " + tse + " %\t |    " + ti + " ms\t |     " + star + "\t|" + disp + " " + str(
            diff))
    print("\t\t|________|_____________|_________|_______________|_______________|_______________|______________|")


# -----------------------------------------------------------------------------------------------------------------------------------------
def Print_DT_Table(r, d, tre, tse, ti, tmp):
    t.sleep(prt_tbl_delay)
    disp = ''
    star = ''
    diff = round(tre - tse, 5)
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    if diff == tmp:
        disp = ' BEST'
    d = str(d)
    tre = str(tre)
    tse = str(tse)
    ti = str(ti)
    repe = str(r)
    if r == 1:
        print("\n\n\n")
        print("\t\t        _------------------------------------------------------------------------------_")
        print("\t\t       |                            Performance Training Table                          |")
        print("\t\t ______|________________________________________________________________________________|_______")
        print("\t\t|   ##   |    D.TREE   |Parameter|  Train Error  |   Test Error  |  Train Time   |    Status    |")
        print("\t\t|________|_____________|_________|_______________|_______________|_______________|______________|")
    print(
        "\t\t|   " + repe + "\t | Performance | Dep=" + d + "\t |    " + tre + " %\t |    " + tse + " %\t |    " + ti + " ms\t |     " + star + "\t|" + disp + " " + str(
            diff))
    print("\t\t|________|_____________|_________|_______________|_______________|_______________|______________|")


# ----------------------------------------------------------------------------------------------------------------------------------------------
main_directory = "E:/981/DataMiningU/Classification-Projectam/"
csv_dir = "csv/"
# csv_file = "car-eval-data-1.csv"
csv_file = "book-evaluation-complete.csv"
# csv_file = "car-eval-data-1.csv"
# csv_file = "plant-classification.csv"
# csv_file = "trn.csv"
# csv_file = "wifi-localization.csv"
path = main_directory + csv_dir + csv_file

Class_OBJECT = Classification()

features_columns = 6
Test_ratio = 0.2

func_delay = 2
slide_delay = 0.8

precision = 5

svm_G_List, svm_C_List, svm_TrainingError_List, svm_TestError_List, svm_Time_List, svm_err_diff, svm_accuracy_diff = [], [], [], [], [], [], []
knn_K_List, knn_TrainingError_List, knn_TestError_List, knn_Time_List, knn_err_diff, knn_accuracy_diff = [], [], [], [], [], []
dt_d_List, dt_TrainingError_List, dt_TestError_List, dt_Time_List, dt_err_diff, dt_accuracy_diff = [], [], [], [], [], []
nb_TrainigError_List, nb_TestError_List, nb_Time_List, nb_err_diff, nb_accuracy_diff = [], [], [], [], []


# ---------------------------------------------------------------------------------------------------------------------------------
def best_parameters(c, g, k, d):
    # c = 0.01
    # g = 0.01
    tround = 1
    for q in range(10):
        for t in range(10):
            print("\t\t\t\tTraining Round " + str(tround))
            autorun(c, g, k, d, True)
            tround += 1
            c += round(rnd.random() + 0.01, 2)
            k = rnd.randint(2, 20)
            d = rnd.randint(1, 20)
        g += round(rnd.random() + 0.01, 2)
    index = 0
    temp = mindiff2(svm_err_diff)
    # print("Temp  : ",temp)
    y = 0
    for y in range(len(svm_C_List)):
        # print("list : ",str(svm_TrainingError_List[y] - svm_TestError_List[y]))
        if round(svm_TrainingError_List[y] - svm_TestError_List[y], precision) == temp:
            # print("girdim ", str(round(svm_TrainingError_List[y] - svm_TestError_List[y], precision)))
            index = y
    best_c = svm_C_List[index]
    best_g = svm_G_List[index]
    index = 0
    temp = mindiff2(knn_err_diff)
    # print("Temp  : ",temp)
    y = 0
    for y in range(len(knn_K_List)):
        # print("list : ",str(svm_TrainingError_List[y] - svm_TestError_List[y]))
        if round(knn_TrainingError_List[y] - knn_TestError_List[y], precision) == temp:
            # print("girdim")
            index = y
    best_k = knn_K_List[index]
    index = 0
    temp = mindiff2(dt_err_diff)
    # print("Temp  : ",temp)
    y = 0
    for y in range(len(dt_d_List)):
        # print("list : ",str(svm_TrainingError_List[y] - svm_TestError_List[y]))
        if round(dt_TrainingError_List[y] - dt_TestError_List[y], precision) == temp:
            # print("girdim")
            index = y
    best_d = dt_d_List[index]
    print("\n\n\n\t [ + ] Gamma\t\t", best_g)
    print("\n\t [ + ] C\t\t", best_c)
    print("\n\t [ + ] K\t\t", best_k)
    print("\n\t [ + ] Depth\t\t", best_d)
    # print(svm_accuracy_diff)
    return best_g, best_c, best_k, best_d


# -------------------------------------------------------------------------------------------------------------------------------------
def autorun(c, g, k, d, repeat_flag):
    os.system('cls')
    Training_Features, Training_Labels, Test_Features, Test_Labels, Features, Labels = Class_OBJECT.buildData(path,
                                                                                                              features_columns,
                                                                                                              Test_ratio)
    total_items = Training_Labels.size + Test_Labels.size
    # print("\t\t\t\t " + "-" * 57)
    # print("\t\t\t\t| Total Items: " + str(total_items) + 
    #         ", Training Items: " + str(Training_Labels.size) +
    #          ", Test Items: " + str(Test_Labels.size) + "|")
    print("\t\t\t\t " + "-" * 57)
    # -------------------------------------------------------------------------------------
    # svm
    time1 = time.now()
    Training_Accuracy, Test_Accuracy, c, g = Class_OBJECT.svm_Classifiction(c, g, Training_Features, Training_Labels,
                                                                            Test_Features, Test_Labels)
    Training_Error = round((1 - Training_Accuracy) * 100, precision)
    Test_Error = round((1 - Test_Accuracy) * 100, precision)
    Training_Accuracy = round(Training_Accuracy * 100, precision)
    Test_Accuracy = round(Test_Accuracy * 100, precision)
    c = round(c, 2)
    g = round(g, 2)
    diff = round(Training_Error - Test_Error, precision)
    star = ''
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    # print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t| [ + ] SVM \t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tC\t\t\t", c, "\t\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tGamma\t\t\t", g, "\t\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTraining Accuracy\t", Training_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTest Accuracy\t\t", Test_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    time2 = time.now()
    total = time2 - time1
    sec = int(total.total_seconds() * 1000)
    print("\t\t\t\t|\t\tTraining Time\t\t", sec, "\tms\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tStatus\t\t\t" + star, "\t\t  |")
    # print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t " + "-" * 57)
    svm_C_List.append(c)
    svm_G_List.append(g)
    svm_TrainingError_List.append(Training_Error)
    svm_TestError_List.append(Test_Error)
    svm_Time_List.append(sec)
    svm_err_diff.append(diff)
    acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
    svm_accuracy_diff.append(acc_diff)
    c += round(rnd.random() + 0.01, 2)
    g += round(rnd.random() + 0.01, 2)
    # ---------------------------------------------------------------------------------
    # n.b
    time1 = time.now()
    Training_Accuracy, Test_Accuracy = Class_OBJECT.naive_bayes_Classifiction(Training_Features, Training_Labels,
                                                                              Test_Features, Test_Labels)
    Training_Error = round((1 - Training_Accuracy) * 100, precision)
    Test_Error = round((1 - Test_Accuracy) * 100, precision)
    Training_Accuracy = round(Training_Accuracy * 100, precision)
    Test_Accuracy = round(Test_Accuracy * 100, precision)
    diff = round(Training_Error - Test_Error, precision)
    star = ''
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    print("\t\t\t\t| [ + ] N.B \t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTraining Accuracy\t", Training_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTest Accuracy\t\t", Test_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    time2 = time.now()
    total = time2 - time1
    sec = int(total.total_seconds() * 1000)
    print("\t\t\t\t|\t\tTraining Time\t\t", sec, "\tms\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tStatus\t\t\t" + star, "\t\t  |")
    print("\t\t\t\t " + "-" * 57)
    nb_TrainigError_List.append(Training_Error)
    nb_TestError_List.append(Test_Error)
    nb_Time_List.append(sec)
    nb_err_diff.append(diff)
    acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
    nb_accuracy_diff.append(acc_diff)
    # -----------------------------------------------------------------------------------------
    # knn
    time1 = time.now()
    Training_Accuracy, Test_Accuracy, ki = Class_OBJECT.knn_Classifiction(k, Training_Features, Training_Labels,
                                                                          Test_Features, Test_Labels)
    Training_Error = round((1 - Training_Accuracy) * 100, precision)
    Test_Error = round((1 - Test_Accuracy) * 100, precision)
    Training_Accuracy = round(Training_Accuracy * 100, precision)
    Test_Accuracy = round(Test_Accuracy * 100, precision)
    diff = round(Training_Error - Test_Error, precision)
    star = ''
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    print("\t\t\t\t| [ + ] KNN \t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tk\t\t\t", ki, "\t\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTraining Accuracy\t", Training_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTest Accuracy\t\t", Test_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    time2 = time.now()
    total = time2 - time1
    sec = int(total.total_seconds() * 1000)
    print("\t\t\t\t|\t\tTraining Time\t\t", sec, "\tms\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tStatus\t\t\t" + star, "\t\t  |")
    print("\t\t\t\t " + "-" * 57)
    knn_K_List.append(ki)
    knn_TrainingError_List.append(Training_Error)
    knn_TestError_List.append(Test_Error)
    knn_Time_List.append(sec)
    knn_err_diff.append(diff)
    acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
    knn_accuracy_diff.append(acc_diff)
    k = rnd.randint(2, 20)
    # ---------------------------------------------------------------------------------------------------------
    # dt
    time1 = time.now()
    Training_Accuracy, Test_Accuracy, depth = Class_OBJECT.Decision_TREE_Classifiction(d, Training_Features,
                                                                                       Training_Labels, Test_Features,
                                                                                       Test_Labels)
    Training_Error = round((1 - Training_Accuracy) * 100, precision)
    Test_Error = round((1 - Test_Accuracy) * 100, precision)
    Training_Accuracy = round(Training_Accuracy * 100, precision)
    Test_Accuracy = round(Test_Accuracy * 100, precision)
    diff = round(Training_Error - Test_Error, precision)
    star = ''
    if diff <= 0 and diff >= -10:
        star = 'GOOD'
    elif diff >= 0 and diff <= 10:
        star = 'Well'
    else:
        star = 'OVERFIT'
    print("\t\t\t\t| [ + ] D.T \t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tDepth\t\t\t", depth, "\t\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTraining Accuracy\t", Training_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tTest Accuracy\t\t", Test_Accuracy, "%", "\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    time2 = time.now()
    total = time2 - time1
    sec = int(total.total_seconds() * 1000)
    print("\t\t\t\t|\t\tTraining Time\t\t", sec, "\tms\t  |")
    print("\t\t\t\t|\t\t\t\t\t\t\t  |")
    print("\t\t\t\t|\t\tStatus\t\t\t" + star, "\t\t  |")
    print("\t\t\t\t" + "-" * 59)
    dt_d_List.append(depth)
    dt_TrainingError_List.append(Training_Error)
    dt_TestError_List.append(Test_Error)
    dt_Time_List.append(sec)
    dt_err_diff.append(diff)
    acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
    dt_accuracy_diff.append(acc_diff)
    d = rnd.randint(1, 20)
    # ---------------------------------------------------------------------------------------------#
    # kmeans
    # time1 = time.now()
    # Accuracy , ki = Class_OBJECT.kmeans_Clustering(km , Features, Labels)
    # Accuracy = round(Accuracy * 100 , precision)
    # print ("\t\t|\n\t\t| [ + ] KMEANS ")
    # print(" \t\t|\t\tk: \t\t\t", ki)
    # print(" \t\t|\t\tAccuracy: \t\t", Accuracy , "%")
    # time2 = time.now()
    # total = time2 - time1
    # sec = int(total.total_seconds() * 1000)
    # print (" \t\t|\t\tTraining Time : \t", sec , " ms")
    # print("\t\t " + "-" * 57)
    if repeat_flag == False:
        input("\n\nPress Any Key To Go Main Menu  . . . \t")
    # km += rnd.randint(2 , 20)
    os.system('cls')
    # ----------------------------------------------------------


# -------------------------------------------------------------------------------------------------------
def mindiff(List):
    List_Count = len(List)
    # print("list count : ",List_Count)
    # print("err list : ",List)
    i1, i2 = 0, 0
    # sort list bigger to smaller
    for i1 in range(List_Count):
        for i2 in range(List_Count - i1 - 1):
            if List[i2] <= List[i2 + 1]:
                List[i2], List[i2 + 1] = List[i2 + 1], List[i2]
    # print("err list : ",List)
    temp = 0
    j = 0
    for j in range(List_Count):
        if min(List) <= 0:
            if List[j] <= 0:
                temp = List[j]
                break
        else:
            temp = min(List)
            break
    # print("temp : ",temp)
    return temp


# -------------------------------------------------------------------------------------------------------
def mindiff2(List):
    List_Count = len(List)
    # print("list count : ",List_Count)
    # print("err list : ",List)
    i1, i2 = 0, 0
    # sort list bigger to smaller
    for i1 in range(List_Count):
        for i2 in range(List_Count - i1 - 1):
            if List[i2] >= List[i2 + 1]:
                List[i2], List[i2 + 1] = List[i2 + 1], List[i2]
    # print("err list : ",List)
    temp = 0
    j = 0
    for j in range(List_Count):
        if List[j] >= -10 and List[j] <= 10:
            if max(List) >= 0:
                if List[j] >= 0:
                    temp = List[j]
                    break
            else:
                temp = max(List)
                break
    # print("temp : ",temp)
    return temp


# -------------------------------------------------------------------------------------------------------
def Menu_check(select, i, flag):
    inputs = "12345678"
    if select not in inputs or len(select) == 0:
        if i == 0:
            os.system('cls')
            print("\n\n\n\n\n\t\t\t\t[ + ] Please Enter a Valid Number . . . ")
            t.sleep(3)
            Menu(c, g, k, d, km)
        else:
            os.system('cls')
            print("\n\n\n\n\n\t\t\t\t[ + ] Number You are Selected Not valid . . . ")
            print("\n\n\n\t\t\t\t[ + ] We Are Routing You To Read File Section . . . ")
            t.sleep(3)
            select = '1'
            flag = True
    elif select != '1' and i == 0:
        if select != '8':
            if select != '7':
                os.system('cls')
                print("\n\n\n\n\n\t\t\t\t[ + ] We Are Routing You To Read File First . . . ")
                t.sleep(3)
                select = '1'


# ----------------------------------------------------------------------------------------------------
def Menu(c, g, k, d):
    os.system('cls')
    # c = 1
    # g = 0.1
    i = 0
    # k = 2
    # d = 1
    r = 1
    flag = False
    while True:
        Print_Menu()
        select = input("\n\t\t\t\t\t Select a Number :\t")
        # Menu_check(select,i,flag)
        os.system('cls')
        # --------------------------------------------------------------------------------------------
        if select == '1':
            Print_CSV(func_delay)
            print("\n ----------------------------------------- ")
            print("| \t Data Set Information\t\t  |")
            print(" ----------------------------------------- ")
            print("\n [ + ] Csv Data Set File Name\t\t" + csv_file)
            print("\n [ + ] Number Of Features Columns\t", features_columns)
            print("\n [ + ] Percent Of Test Data\t\t", Test_ratio * 100, " %")

            Training_Features, Training_Labels, Test_Features, Test_Labels, All_Features, All_Labels = Class_OBJECT.buildData(
                path, features_columns, Test_ratio)
            lines = Training_Labels.size + Test_Labels.size
            print("\n [ + ] Imported Lines\t\t\t", lines)
            print("\n [ + ] Training Items\t\t\t", Training_Labels.size)
            print("\n [ + ] Test Items\t\t\t", Test_Labels.size)
            if flag == False:
                input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
            i += 1
        # --------------------------------------------------------------------------------------------
        elif select == '2':
            Print_SVM(func_delay)
            print(" --------------------------------------------- ")
            print("| Support Vector Machine (SVM) Classification |")
            print(" --------------------------------------------- ")
            time1 = time.now()
            Training_Accuracy, Test_Accuracy, c, g = Class_OBJECT.svm_Classifiction(c, g, Training_Features,
                                                                                    Training_Labels, Test_Features,
                                                                                    Test_Labels)
            c = round(c, 2)
            g = round(g, 2)
            print("\n [ /*\ ] 1 . Parameters")
            print("\n  [ + ] C\t\t\t", c)
            print("\n  [ + ] Gamma\t\t\t", g)
            Training_Accuracy = round(Training_Accuracy * 100, precision)
            Test_Accuracy = round(Test_Accuracy * 100, precision)
            Accuracy_diff = round(Training_Accuracy - Test_Accuracy, precision)
            Training_Error = round(100 - Training_Accuracy, precision)
            Test_Error = round(100 - Test_Accuracy, precision)
            diff = round(Training_Error - Test_Error, precision)

            print("\n\n [ /*\ ] 2 . Performance")
            print("\n  [ + ] Training Accuracy\t", Training_Accuracy, "%")
            print("\n  [ + ] Test Accuracy\t\t", Test_Accuracy, "%")
            print("\n  [ + ] Accuracy Difference\t", Accuracy_diff, "%")

            print("\n\n [ /*\ ] 3 . Error")
            print("\n  [ + ] Training Error\t\t", Training_Error, "%")
            print("\n  [ + ] Test Error\t\t", Test_Error, "%")
            print("\n  [ + ] Error Difference\t", diff, "%")
            time2 = time.now()
            total = time2 - time1
            sec = int(total.total_seconds() * 1000)
            print("\n\n [ /*\ ] 4 . Time")
            print("\n  [ + ] Training Time\t\t", sec, " MiliSeconds")

            star = ''
            if diff <= 0 and diff >= -10:
                star = 'GOOD'
            elif diff >= 0 and diff <= 10:
                star = 'Well'
            else:
                star = 'OVERFIT'
            print("\n\n [ /*\ ] 5. Status\t\t" + star)
            input("\n\nPress Any Key To Go Table  . . . \t")
            os.system('cls')

            svm_G_List.append(g)
            svm_C_List.append(c)
            svm_TrainingError_List.append(Training_Error)
            svm_TestError_List.append(Test_Error)
            svm_Time_List.append(sec)
            svm_err_diff.append(diff)
            acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
            svm_accuracy_diff.append(acc_diff)
            temp = mindiff2(svm_err_diff)
            p = 0
            for p in range(len(svm_C_List)):
                if p == 0:
                    r = 1
                Print_SVM_Table(r, svm_G_List[p], svm_C_List[p], svm_TrainingError_List[p], svm_TestError_List[p],
                                svm_Time_List[p], temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
            i += 1
            c += round(rnd.random() + 0.01, 2)
            g += round(rnd.random() + 0.01, 2)
        # --------------------------------------------------------------------------------------------
        elif select == '3':
            Print_NB(func_delay)
            print("\n -------------------------------------------- ")
            print("|\tNaive Bayes Classification \t     |")
            print(" -------------------------------------------- ")
            time1 = time.now()
            Training_Accuracy, Test_Accuracy = Class_OBJECT.naive_bayes_Classifiction(Training_Features,
                                                                                      Training_Labels, Test_Features,
                                                                                      Test_Labels)
            Training_Accuracy = round(Training_Accuracy * 100, precision)
            Test_Accuracy = round(Test_Accuracy * 100, precision)
            Accuracy_diff = round(Training_Accuracy - Test_Accuracy, precision)
            Training_Error = round(100 - Training_Accuracy, precision)
            Test_Error = round(100 - Test_Accuracy, precision)
            diff = round(Training_Error - Test_Error, precision)

            print("\n\n [ /*\ ] 1 . Performance")
            print("\n  [ + ] Training Accuracy\t", Training_Accuracy, "%")
            print("\n  [ + ] Test Accuracy\t\t", Test_Accuracy, "%")
            print("\n  [ + ] Accuracy Difference\t", Accuracy_diff, "%")

            print("\n\n [ /*\ ] 2 . Error")
            print("\n  [ + ] Training Error\t\t", Training_Error, "%")
            print("\n  [ + ] Test Error\t\t", Test_Error, "%")
            print("\n  [ + ] Error Difference\t", diff, "%")
            time2 = time.now()
            total = time2 - time1
            sec = int(total.total_seconds() * 1000)
            print("\n\n [ /*\ ] 3 . Time")
            print("\n  [ + ] Training Time\t\t", sec, " MiliSeconds")
            star = ''
            if diff <= 0 and diff >= -10:
                star = 'GOOD'
            elif diff >= 0 and diff <= 10:
                star = 'Well'
            else:
                star = 'OVERFIT'
            print("\n\n [ /*\ ] 5. Status\t\t" + star)
            input("\n\nPress Any Key To Go Table  . . . \t")
            os.system('cls')

            nb_TrainigError_List.append(Training_Error)
            nb_TestError_List.append(Test_Error)
            nb_Time_List.append(sec)
            nb_err_diff.append(diff)
            acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
            nb_accuracy_diff.append(acc_diff)
            temp = mindiff2(nb_err_diff)
            p = 0
            for p in range(len(nb_TrainigError_List)):
                if p == 0:
                    r = 1
                Print_NB_Table(r, nb_TrainigError_List[p], nb_TestError_List[p], nb_Time_List[p], temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
            i += 1
        # --------------------------------------------------------------------------------------------
        elif select == '4':
            Print_KNN(func_delay)
            print("\n -------------------------------------------- ")
            print("|  KNN (k-Nearest Neighbors) Classification  |")
            print(" -------------------------------------------- ")
            time1 = time.now()
            Training_Accuracy, Test_Accuracy, ki = Class_OBJECT.knn_Classifiction(k, Training_Features, Training_Labels,
                                                                                  Test_Features, Test_Labels)
            print("\n [ /*\ ] 1 . Parameters")
            print("\n  [ + ] k\t\t\t", ki)
            Training_Accuracy = round(Training_Accuracy * 100, precision)
            Test_Accuracy = round(Test_Accuracy * 100, precision)
            Accuracy_diff = round(Training_Accuracy - Test_Accuracy, precision)
            Training_Error = round(100 - Training_Accuracy, precision)
            Test_Error = round(100 - Test_Accuracy, precision)
            diff = round(Training_Error - Test_Error, precision)

            print("\n\n [ /*\ ] 2 . Performance")
            print("\n  [ + ] Training Accuracy\t", Training_Accuracy, "%")
            print("\n  [ + ] Test Accuracy\t\t", Test_Accuracy, "%")
            print("\n  [ + ] Accuracy Difference\t", Accuracy_diff, "%")

            print("\n\n [ /*\ ] 3 . Error")
            print("\n  [ + ] Training Error\t\t", Training_Error, "%")
            print("\n  [ + ] Test Error\t\t", Test_Error, "%")
            print("\n  [ + ] Error Difference\t", diff, "%")
            time2 = time.now()
            total = time2 - time1
            sec = int(total.total_seconds() * 1000)
            print("\n\n [ /*\ ] 4 . Time")
            print("\n  [ + ] Training Time\t\t", sec, " MiliSeconds")
            star = ''
            if diff <= 0 and diff >= -10:
                star = 'GOOD'
            elif diff >= 0 and diff <= 10:
                star = 'Well'
            else:
                star = 'OVERFIT'
            print("\n\n [ /*\ ] 5. Status\t\t" + star)
            input("\n\nPress Any Key To Go Table  . . . \t")
            os.system('cls')
            knn_K_List.append(ki)
            knn_TrainingError_List.append(Training_Error)
            knn_TestError_List.append(Test_Error)
            knn_Time_List.append(sec)
            knn_err_diff.append(diff)
            acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
            knn_accuracy_diff.append(acc_diff)
            temp = mindiff2(knn_err_diff)
            p = 0
            for p in range(len(knn_K_List)):
                if p == 0:
                    r = 1
                Print_KNN_Table(r, knn_K_List[p], knn_TrainingError_List[p], knn_TestError_List[p], knn_Time_List[p],
                                temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
            i += 1
            k = rnd.randint(2, 20)
        # --------------------------------------------------------------------------------------------
        elif select == '5':
            Print_DT(func_delay)
            print("\n -------------------------------------------- ")
            print("|\tDecision Tree Classification\t     |")
            print(" -------------------------------------------- ")
            time1 = time.now()
            Training_Accuracy, Test_Accuracy, depth = Class_OBJECT.Decision_TREE_Classifiction(d, Training_Features,
                                                                                               Training_Labels,
                                                                                               Test_Features,
                                                                                               Test_Labels)
            print("\n [ /*\ ] 1 . Parameters")
            print("\n  [ + ] Depth\t\t\t", depth)
            Training_Accuracy = round(Training_Accuracy * 100, precision)
            Test_Accuracy = round(Test_Accuracy * 100, precision)
            Accuracy_diff = round(Training_Accuracy - Test_Accuracy, precision)
            Training_Error = round(100 - Training_Accuracy, precision)
            Test_Error = round(100 - Test_Accuracy, precision)
            diff = round(Training_Error - Test_Error, precision)

            print("\n\n [ /*\ ] 2 . Performance")
            print("\n  [ + ] Training Accuracy\t", Training_Accuracy, "%")
            print("\n  [ + ] Test Accuracy\t\t", Test_Accuracy, "%")
            print("\n  [ + ] Accuracy Difference\t", Accuracy_diff, "%")

            print("\n\n [ /*\ ] 3 . Error")
            print("\n  [ + ] Training Error\t\t", Training_Error, "%")
            print("\n  [ + ] Test Error\t\t", Test_Error, "%")
            print("\n  [ + ] Error Difference\t", diff, "%")
            time2 = time.now()
            total = time2 - time1
            sec = int(total.total_seconds() * 1000)
            print("\n\n [ /*\ ] 4 . Time")
            print("\n  [ + ] Training Time\t\t", sec, " MiliSeconds")
            star = ''
            if diff <= 0 and diff >= -10:
                star = 'GOOD'
            elif diff >= 0 and diff <= 10:
                star = 'Well'
            else:
                star = 'OVERFIT'
            print("\n\n [ /*\ ] 5. Status\t\t" + star)
            input("\n\nPress Any Key To Go Table  . . . \t")
            os.system('cls')
            dt_d_List.append(depth)
            dt_TrainingError_List.append(Training_Error)
            dt_TestError_List.append(Test_Error)
            dt_Time_List.append(sec)
            dt_err_diff.append(diff)
            acc_diff = round(Training_Accuracy - Test_Accuracy, precision)
            dt_accuracy_diff.append(acc_diff)
            temp = mindiff2(dt_err_diff)
            p = 0
            for p in range(len(dt_d_List)):
                if p == 0:
                    r = 1
                Print_DT_Table(r, dt_d_List[p], dt_TrainingError_List[p], dt_TestError_List[p], dt_Time_List[p], temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
            i += 1
            d = rnd.randint(1, 20)
        # --------------------------------------------------------------------------------------------
        else:
            # exit()
            main_Menu(c, g, k, d, km)
        # --------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------
def PrintTables():
    os.system('cls')
    # --------------------------------------------------------------------------------------------
    if len(svm_C_List) == 0 or len(nb_TrainigError_List) == 0 or len(knn_K_List) == 0 or len(dt_d_List) == 0:
        os.system('cls')
        print("\n\n\n\n\t\t\t\t[ + ] One of Given Lists is Empty Please Run Classifier First . . . ")
        t.sleep(2)
        os.system('cls')
        main_Menu(c, g, k, d, km)
    # --------------------------------------------------------------------------------------------
    while True:
        Print_Table_Menu()
        select = input("\n\n\t\t\t\t\tSelection : \t")
        # --------------------------------------------------------------------------------------------
        if select == '1':
            os.system('cls')
            temp = mindiff2(svm_err_diff)
            p = 0
            for p in range(len(svm_C_List)):
                if p == 0:
                    r = 1
                Print_SVM_Table(r, svm_G_List[p], svm_C_List[p], svm_TrainingError_List[p], svm_TestError_List[p],
                                svm_Time_List[p], temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
        # --------------------------------------------------------------------------------------------
        elif select == '2':
            os.system('cls')
            temp = mindiff2(nb_err_diff)
            p = 0
            for p in range(len(nb_TrainigError_List)):
                if p == 0:
                    r = 1
                Print_NB_Table(r, nb_TrainigError_List[p], nb_TestError_List[p], nb_Time_List[p], temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
        # --------------------------------------------------------------------------------------------
        elif select == '3':
            os.system('cls')
            temp = mindiff2(knn_err_diff)
            p = 0
            for p in range(len(knn_K_List)):
                if p == 0:
                    r = 1
                Print_KNN_Table(r, knn_K_List[p], knn_TrainingError_List[p], knn_TestError_List[p], knn_Time_List[p],
                                temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
        # --------------------------------------------------------------------------------------------
        elif select == '4':
            os.system('cls')
            temp = mindiff2(dt_err_diff)
            p = 0
            for p in range(len(dt_d_List)):
                if p == 0:
                    r = 1
                Print_DT_Table(r, dt_d_List[p], dt_TrainingError_List[p], dt_TestError_List[p], dt_Time_List[p], temp)
                r += 1
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
        # --------------------------------------------------------------------------------------------
        else:
            main_Menu(c, g, k, d, km)
        # --------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------
def main_Menu(c, g, k, d, km):
    os.system('cls')
    while True:
        Print_Main_Menu()
        select = input("\n\n\t\t\t\t\tSelection : \t")
        # --------------------------------------------------------------------------------------------
        if select == '1':
            Menu(c, g, k, d)
            c += round(rnd.random() + 0.01, 2)
            g += round(rnd.random() + 0.01, 2)
            k = rnd.randint(2, 20)
            d = rnd.randint(1, 20)
        # --------------------------------------------------------------------------------------------
        elif select == '2':
            autorun(c, g, k, d, False)
            c += round(rnd.random() + 0.01, 2)
            g += round(rnd.random() + 0.01, 2)
            k = rnd.randint(2, 20)
            d = rnd.randint(1, 20)
        # --------------------------------------------------------------------------------------------
        elif select == '3':
            PrintTables()
        # --------------------------------------------------------------------------------------------
        elif select == '4':
            os.system('cls')
            rep = int(input("\n\n\n\n\t\t\t\t [ + ] Enter number of Repeats of Classifiers . . .  : \t"))
            for h in range(rep):
                print("\t\t\t\tTraining Round " + str(h))
                autorun(c, g, k, d, True)
                c += round(rnd.random() + 0.01, 2)
                g += round(rnd.random() + 0.01, 2)
                k = rnd.randint(2, 20)
                d = rnd.randint(1, 20)
                # t.sleep(0.01)
        # --------------------------------------------------------------------------------------------
        elif select == '5':
            os.system('cls')
            time1 = time.now()
            best_g, best_c, best_k, best_d = best_parameters(c, g, k, d)
            # c = round(c , 2)
            # g = round(g , 2)
            file = open(main_directory + "Best_Parameters.txt", "a+")
            file.write("Gamma : " + str(best_g) + "\t\t,C : " + str(best_c) + "\t\t,K : " + str(
                best_k) + "\t\t,Depth : " + str(best_d) + "\n")
            file.close()
            time2 = time.now()
            total = time2 - time1
            sec = int(total.total_seconds())
            print("\t\n [ + ] Total Training Time : \t", sec, " sec")
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
        # --------------------------------------------------------------------------------------------
        elif select == '6':
            os.system('cls')
            Print_KMEANS(func_delay)
            print("\n -------------------------------------------- ")
            print("|\t      KMEANS Clustring\t\t     |")
            print(" -------------------------------------------- ")
            time1 = time.now()
            input("\n [ + ] Press Any Key To Building Data From Csv Data Set File . . . \t")
            Training_Features, Training_Labels, Test_Features, Test_Labels, All_Features, All_Labels = Class_OBJECT.buildData(
                path, features_columns, Test_ratio)
            lines = Training_Labels.size + Test_Labels.size
            print("\n [ + ] Imported Lines\t\t\t", lines)
            print("\n [ + ] Training Items\t\t\t", Training_Labels.size)
            print("\n [ + ] Test Items\t\t\t", Test_Labels.size)
            input("\n [ + ] Press Any Key To Plot Showing . . . ")
            accuracy, ki = Class_OBJECT.kmeans_Clustering(km, All_Features, All_Labels)
            print("\n [ + ] k\t\t", ki)
            print("\n [ + ] Accuracy\t\t", round(accuracy * 100, 2), "%")
            print("\n [ + ] Error\t\t", round((1 - accuracy) * 100, 2), "%")
            time2 = time.now()
            total = time2 - time1
            sec = round(total.total_seconds(), precision)
            print("\n [ + ] Training Time\t", sec, " Seconds")
            input("\n\nPress Any Key To Go Menu  . . . \t")
            os.system('cls')
            km += rnd.randint(2, 20)
        # --------------------------------------------------------------------------------------------
        elif select == '7':
            os.system('cls')
            sd = slide_delay
            while True:
                Print_CSV(sd)
                Print_SVM(sd)
                Print_NB(sd)
                Print_KNN(sd)
                Print_DT(sd)
                Print_KMEANS(sd)
            i += 1
        # --------------------------------------------------------------------------------------------
        else:
            os.system('cls')
            exit()
        # --------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # c = 1
    # g = 0.1
    # k = 2
    # d = 1
    # km = 2
    c = round(rnd.random() + 0.01, 2)
    g = round(rnd.random() + 0.01, 2)
    k = rnd.randint(2, 20)
    d = rnd.randint(1, 20)
    km = rnd.randint(2, 20)
    main_Menu(c, g, k, d, km)
# ---------------------------------------------------------------------------------------------------------------------------------
