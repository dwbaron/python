"""
This is a demo of auto-factory programming schema
all the cars' orders arrive at the time follow a specific distribution
there are still a lot functions remain to be implemented
"""

from collections import defaultdict
import random
import numpy as np
from scipy import stats
# current dict of all cars created by now
curr_dict = defaultdict(list)

class Car:
    def __init__(self, name):
        self.name = name
        curr_dict[name].append(self)

    def run(self):
        print('   ________  \n'
              ' _/        | \n'
              '|__--__--__| \n'
              '  (_)  (_)   \n')


# different types of cars of Ranger Rover
class Rover(Car):
    def __init__(self):
        Car.__init__(self, name='Rover')
        # crossing performance measured in meters
        self.cross = 1
        # four-wheel drive
        self.qua = True
        self.price = 200

    def run(self):
        print('      .------------.  \n'
              ' ____/__|   [__]    \ \n'
              '/ |||               | \n'
              '|__.**.______ .**._.* \n'
              '   *..*       *..*    \n')


class Discovery(Rover):
    def __init__(self):
        Car.__init__(self, name='Discovery')
        self.cross = 1
        self.qua = False
        self.price = 80

    def run(self):
        print('     .-----------. \n'
              ' ___/__|  [__] | | \n'
              '/              | | \n'
              '|_ .**.____.**.|_| \n'
              '   *..*    *..*    \n')


class Aurora(Rover):
    def __init__(self):
        Car.__init__(self, name='Aurora')
        self.cross = 0.5
        self.qua = False
        self.price = 60

    def run(self):
        print('    ,---------.  \n'
              '.../           | \n'
              '|_ .**.___.**._/ \n'
              '   *..*   *..*   \n')


class Sport(Rover):
    def __init__(self):
        Car.__init__(self, name='Sport')
        self.cross = 1
        self.qua = True
        self.price = 120

    def run(self):
        print('     ------------.  \n'
              '  __/__|  [__]    | \n'
              '/ ##              | \n'
              'o_.**._____ .**._.* \n'
              '  *..*      *..*    \n')

# global list of cars classes
car_list = (Rover, Aurora, Discovery, Sport)


# Initialize state
# Return a tuple : (car_names, car_dict)
def init_cars(c_list):
    car_names = []
    for car in c_list:
        car_names.append(car.__name__)
    car_dict = dict(zip(car_names, c_list))
    return car_names, car_dict

# global states
cars_names, cars_dict = init_cars(car_list)


# auto-car creation factory receive the results from init_cars()
def car_factory(name, names=cars_names, cars=cars_dict):
    if name in names:
        # create a new car
        cls = cars[name]
        # init a car instance and add to curr_dict
        new_car = cls()
    else:
        print('Wrong car type')
        return
    return new_car


# a car store class
class CarStore:
    def __init__(self, name):
        self.name = name
        self.bills = []
        self.income = 0

    def order(self):
        # any order bills
        if self.bills:
            new_car = car_factory(self.bills.pop())
            return new_car

    def upgrade(self):
        pass


def simulation(expo=1, flag=False):
    max_bills = 100
    # generate bills
    store = CarStore('rover')
    for i in range(max_bills):
        # generate random bills
        car = random.choice(cars_names)
        store.bills.append(car)

    # simulate cars arrival
    time = 0
    lambd = 0.3

    # according to exponential dist
    if expo == 1:
        time_interval = stats.expon.rvs(1/lambd, size=max_bills)
        arrival_time = np.round(np.cumsum(time_interval))
    # else according to uniform dist
    elif expo == 2:
        time_interval = [random.randint(1, 11) for _ in range(max_bills)]
        arrival_time = np.round(np.cumsum(time_interval))

    # gauss dist
    elif expo == 3:
        time_interval = [random.gauss(6, 2) for _ in range(max_bills)]
        while min(time_interval) <= 0:
            time_interval = [random.gauss(6, 2) for _ in range(max_bills)]
        arrival_time = np.round(np.cumsum(time_interval))
    price_series = []

    # simulation
    while time < 1000:

        # meet a car's bill
        if time in arrival_time:
            # create a car
            car = store.order()
            # update income
            store.income += car.price
            print('At Time {:<2} New {} arrival:'.format(time, car.name))
            # show car
            car.run()
            # collect price for latter image plot
            price_series.append(car.price)
        else:
            pass
        time += 1
        if time > max(arrival_time):
            break

    # show result
    for k, v in curr_dict.items():
        print('{:<10}:{:<2}  income:{}'.format(k, len(v), sum([i.price for i in v])))
    # income
    print('total income is {}'.format(store.income))
    # show image
    if flag:
        
        import matplotlib.pyplot as plt
        # an dist hist plot
        plt.subplot(1, 2, 1)
        plt.ylim((0, 50))
        plt.hist(time_interval)
        plt.xlabel('Time Interval')
        plt.ylabel('Frequency')

        # time-price plot
        plt.subplot(1, 2, 2)
        plt.ylim(50, 220)
        plt.plot(arrival_time, price_series, '-^r')
        plt.xlabel('Time')
        plt.ylabel('Income')
        plt.show()

if __name__ == '__main__':
    simulation(expo=2,flag=True)
