import json
import multiprocessing
import random
import time
import traceback


def bubble_sort2(arr):
    n = len(arr)

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break


def recursion3(count_to, start=0):
    if start >= count_to:
        print("Finish")
        return
    start += 1
    print(start)
    return recursion3(count_to, start)


def lambdas_etc4():
    def create_sum(num):
        return lambda x: x + num

    sum = create_sum(10)
    print(sum(5))

    class Static_method_holder:
        @staticmethod
        def st_methd(message):
            print(message)

        def non_static_method(self, message):
            print(message)

    # Static_method_holder.non_static_method("message")
    Static_method_holder.st_methd("message")

    def delegate(x):
        return x + 10

    def delegat_user(delegate, x):
        print(delegate(x))

    delegat_user(delegate, 10)
    delegat_user(lambda x: x + 10, 10)


def timer_bubble6():
    def timer_wrapper(delegate):
        def timed(**kwargs):
            now = time.time()
            delegate(**kwargs)
            end = time.time()
            elapsed = end - now
            print(elapsed)

        return timed

    @timer_wrapper
    def bubble(arr):
        n = len(arr)

        for i in range(n):
            swap = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swap = True
            if not swap:
                break

    bubble(arr=[1, 2, 3, 40, 5, 123, 24])


def operators7():
    def decorator(func):
        def decorated():
            print("Decorated")
            func()

        return decorated

    @decorator
    def decorated_proc():
        print("Called whats inside")

    def proc(x: 'int', y: 'str') -> 'list':
        pass

    print(proc.__annotations__)
    decorated_proc()

    class Operators:
        def __init__(self, message):
            self.message = message

        def __add__(self, other):
            self.message = self.message + other.message

    op1 = Operators("Hello ")
    op2 = Operators("World!")
    op1 + op2
    print(op1.message)


def inheritence_etc8():
    class Parent:
        def parent_func(self):
            print("Am parent")

    class Child(Parent):
        def parent_func(self):
            print("Am child")

    p = Parent()
    c = Child()
    p.parent_func()
    c.parent_func()


def integrity9():
    def func(someinput: 'int'):
        if not isinstance(someinput, int):
            raise ValueError("Not int")
        print(someinput)

    func(1)
    func("my balls")


def database_coms10():
    class Connection:
        """
        Connects to database, sends queries
        """
        connection = None

        # make singleton
        def __init__(self, config):
            # read config
            pass

        def connect(self):
            # connect to the database
            # save connection to a variable
            # TODO: Exception handling
            pass

        def query(self, query, args):
            """
            Query the database
            :param query:
            :param args: Arguments for query
            :return: Response
            """
            # Create cursor
            # Executre query
            # Return response

    class UserTable:
        """
        Handles CRUD of user table
        """

        # Make singleton
        def __init__(self):
            # Define CRUD SQL syntax
            pass

        def create(self, instance):
            # Map instance to sql args for CRUD
            # Send request to connection via query
            # Return result or new id
            # TODO: Except mishaps in instance
            pass

        def read(self, args):
            # Map args to request
            # Send request
            # Return data from database in array
            # TODO: Except error where id doesnt exist or is wrong
            pass

        def update(self, instance):
            # Map instance to request
            # Send request
            # Return result True False
            # TODO: Except error where id doesnt exist or is wrong
            pass

        def delete(self, instance):
            # Get id of instance
            # Map id to request
            # Send request
            # Return result True False
            # TODO: Except error where id doesnt exist
            pass

    class UserRow:
        """
        Allows to manipulate with data from db
        """

        def __init__(self):
            # Instance filling
            pass

        def map_to_req(self):
            """
            Maps data from this instance to request
            :return: Query args
            """

        def map_from_answer(self, data):
            """
            Maps data from db to this instance
            """


def socket_comms11():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("Address", "port"))
        sock.listen()
        while True:
            conn, addr = sock.accept()
            while True:
                data = conn.recv(1024).decode("ascii")
                send = "Some message"
                conn.send(send.encode())


def aggregation_composition14():
    class Departament:
        dept_num: int

        def __init__(self, dept_num):
            self.dept_num = dept_num


class Testing16:
    def __init__(self, message: str):
        self.message = message


def serialization16():
    test_inst = Testing16("Hey, i want to be serialized")

    # with open("text.pickle", "bw") as file:
    #     pickle.dump(test_inst, file)
    #     file.close()

    # with open("text.pickle", "br") as file:
    #     loaded = pickle.load(file)
    #     print(loaded)
    #     print(loaded.message)
    #     file.close()

    with open("text.json", "w+") as file:
        dct = test_inst.__dict__
        json.dump(dct, file)
        file.close()

    with open("text.json", "r+") as file:
        dct = json.load(file)
        inst = Testing16("None")
        inst.__dict__.update(dct)
        print(inst.message)
        file.close()


def regresion_class18():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import sklearn.linear_model as lm
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mlcollege/ai-academy/main/06-Regrese/data/ceny_domu.csv", sep=',')
    # Clean dataset
    df.drop_duplicates()
    df.dropna()
    # Format dataset
    # Standartize dataset
    # Split train and test data
    X = "Inputs"
    y = "Answers"
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    # Supervised training
    lr = lm.LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)


def neurons19():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation

    # import data
    df = pd.read_csv("address for data")
    # clean data
    df.dropna(how="all")
    df.drop_duplicates(subset=["collumn x"])
    # format data
    # change FL to Florida etc
    # separate data train test
    input_features = ["names", "of", "collumns", "we want to use for training"]
    target = ["name of column that is target"]
    X_train, X_test, y_train, y_test = train_test_split(df[input_features], df[target], train_size=.75)
    # standardize data
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    # create model
    my_model = Sequential()
    # create hidden layers
    my_model.add(Dense(30, input_shape=4))
    my_model.add(Activation("relu"))
    my_model.add(Dense(1, activation="sigmoid"))
    # create training functions
    my_model.compile(
        loss='mse',
        optimiser='adam',
        metrics=['mse, mae']
    )
    # train model
    my_model.fit(
        X_train_std, y_train,
        batch_size=64,
        epochs=100,
        validation_data=(X_test_std, y_test)
    )
    # test model
    y_pred = my_model.predict(X_test_std)
    print(mean_absolute_error(y_true=y_test, y_pred=y_pred))
    pass


def unitTesting20():
    def sum_it(x, y):
        return x + y


def t1f(holdr, lock):
    lock.acquire()
    try:
        for i in range(50):
            holdr["state"] = holdr["state"] + "a"
    finally:
        lock.release()


def t2f(holdr, lock):
    lock.acquire()
    try:
        for i in range(50):
            holdr["state"] = holdr["state"] + "b"
    finally:
        lock.release()


def processes22():
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()

    holdr = manager.dict()
    holdr["state"] = "s"

    p1 = multiprocessing.Process(target=t1f, args=(holdr, lock,))
    p2 = multiprocessing.Process(target=t2f, args=(holdr, lock,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(holdr)


def producer(q):
    for i in range(10):
        rnd = random.randint(1, 15)
        q.put(rnd)
        print(f"Created {rnd}")
    q.put(None)


def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Read {item}")
        print("Sleeping")
        time.sleep(1)


def produce_consumer22():
    manager = multiprocessing.Manager()
    q = manager.Queue()

    p1 = multiprocessing.Process(target=producer, args=(q,))
    p2 = multiprocessing.Process(target=consumer, args=(q,))
    p1.start()
    p2.start()

    p1.join()
    p2.join()


def data_collection_props23():
    tuple1 = ("b", "a", 1, 1)
    print(tuple1)
    print(tuple1[3])
    list_test = ["b", "a", 2, 2]
    print(list_test)
    print(list_test[0])
    set_test = {"b", "a", "b", 2, 1, 5}
    print(set_test)
    dic = {
        (1, 2, 3): "b",
        "a": "b",
        "c": "b",
        1: 1,
        2: 1
    }
    print(dic)


def assertions24():
    x = 0
    assert x > 1, "balls"
    try:
        y = 0 / 0
    except Exception as e:
        traceback.print_exception(e)
    finally:
        print("You stupid")


def string_manipulations():
    import re

    string = "lorem ipsum hello world, toy story, programming is fun, maturita. nejaky randomni text"
    # print(string.encode("utf-16"))
    # print(string.lower())
    # print(string.upper())
    # print(string.isdigit())
    # print(string.rstrip("t"))
    # print(string.lstrip("L"))
    # print(string.capitalize())
    # print(string.rfind("l"))
    # print(string.split(" "))
    print(re.findall(r"[A-z]*m", string))

    with open("text.txt", "r+") as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip("\n")
            if re.findall(r"z[A-z]d", line):
                print("Found")
            print(line)


def sum_it(x, y):
    return x + y


if __name__ == '__main__':
    integrity9()
