import os

import sys

from config import cfg

batch_size = cfg.batch_size


def unpickle(file):
    if (2, 7) <= sys.version_info < (3,):
        specify_encoding = False
        import cPickle as pickle
    else:
        specify_encoding = True
        import pickle

    with open(file, 'rb') as fo:
        if specify_encoding:
            dict = pickle.load(fo, encoding='bytes')
        else:
            dict = pickle.load(fo)
    return dict


def build_tree():
    terrestrial = []
    terrestrial_natural = []
    terrestrial_artificial = []
    terrestrial_artificial_structure = []
    terrestrial_artificial_structure_building = []

    sea = []

    cloud = []

    flora = []
    flora_tree = []
    flora_plant = []
    flora_plant_fruit = []
    flora_plant_flower = []

    object = []
    object_vehicle = []
    object_vehicle_two = []
    object_vehicle_four = []
    object_vehicle_transport = []
    object_container = []
    object_container_food = []
    object_container_drink = []
    object_electronic = []
    object_furniture = []
    object_furniture_seat = []

    being = []
    being_person = []
    being_person_adult = []
    being_person_child = []
    being_animal = []
    being_animal_aquatic = []
    being_animal_aquatic_mammal = []
    being_animal_aquatic_fish = []
    being_animal_aquatic_fish_el = []
    being_animal_invertebrates = []
    being_animal_invertebrates_insects = []
    being_animal_vertebrates = []
    being_animal_vertebrates_mammals = []
    being_animal_vertebrates_mammals_marsupial = []
    being_animal_vertebrates_mammals_carnivore = []
    being_animal_vertebrates_mammals_carnivore_canid = []
    being_animal_vertebrates_mammals_carnivore_felid = []
    being_animal_vertebrates_mammals_rodent = []
    being_animal_vertebrates_reptiles = []

    train_meta = unpickle(os.path.abspath('cifar-100-python/meta'))
    train_labels_names = train_meta[b'fine_label_names']

    for i in range(len(train_labels_names)):
        name = train_labels_names[i]

        if name in [b'mountain', b'plain']:
            terrestrial_natural.append([i])
        elif name in [b'castle', b'house', b'skyscraper']:
            terrestrial_artificial_structure_building.append([i])
        elif name == b'bridge':
            terrestrial_artificial_structure.append([i])
        elif name == b'road':
            terrestrial_artificial.append([i])

        elif name == b'sea':
            sea.append([i])

        elif name == b'cloud':
            cloud.append([i])

        elif name in [b'apple', b'orange', b'pear', b'sweet_pepper']:
            flora_plant_fruit.append([i])
        elif name in [b'orchid', b'poppy', b'rose', b'sunflower', b'tulip']:
            flora_plant_flower.append([i])
        elif name == b'mushroom':
            flora_plant.append([i])
        elif name in [b'maple_tree', b'oak_tree', b'palm_tree', b'pine_tree', b'willow_tree']:
            flora_tree.append([i])
        elif name == b'forest':
            flora.append([i])

        elif name in [b'bowl', b'plate']:
            object_container_food.append([i])
        elif name in [b'bottle', b'can', b'cup']:
            object_container_drink.append([i])
        elif name in [b'clock', b'keyboard', b'telephone', b'television']:
            object_electronic.append([i])
        elif name in [b'bed', b'table', b'wardrobe']:
            object_furniture.append([i])
        elif name in [b'chair', b'couch']:
            object_furniture_seat.append([i])
        elif name in [b'tank', b'tractor', b'rocket']:
            object_vehicle.append([i])
        elif name in [b'bicycle', b'motorcycle']:
            object_vehicle_two.append([i])
        elif name in [b'pickup_truck', b'streetcar']:
            object_vehicle_four.append([i])
        elif name in [b'bus', b'train']:
            object_vehicle_transport.append([i])
        elif name in [b'lamp', b'lawn_mower']:
            object.append([i])

        elif name in [b'man', b'woman']:
            being_person_adult.append([i])
        elif name in [b'boy', b'girl']:
            being_person_child.append([i])
        elif name in [b'baby']:
            being_person.append([i])
        elif name in [b'beaver', b'porcupine', b'hamster', b'mouse', b'shrew', b'squirrel', b'rabbit']:
            being_animal_vertebrates_mammals_rodent.append([i])
        elif name in [b'camel', b'cattle', b'chimpanzee', b'elephant']:
            being_animal_vertebrates_mammals.append([i])
        elif name in [b'kangaroo', b'possum']:
            being_animal_vertebrates_mammals_marsupial.append([i])
        elif name in [b'crab', b'lobster', b'snail', b'spider', b'worm']:
            being_animal_invertebrates.append([i])
        elif name in [b'dolphin', b'whale', b'otter', b'seal']:
            being_animal_aquatic_mammal.append([i])
        elif name in [b'aquarium_fish', b'flatfish', b'trout']:
            being_animal_aquatic_fish.append([i])
        elif name in [b'bee', b'beetle', b'butterfly', b'caterpillar', b'cockroach']:
            being_animal_invertebrates_insects.append([i])
        elif name in [b'bear', b'raccoon', b'skunk']:
            being_animal_vertebrates_mammals_carnivore.append([i])
        elif name in [b'crocodile', b'dinosaur', b'lizard', b'snake', b'turtle']:
            being_animal_vertebrates_reptiles.append([i])
        elif name in [b'wolf', b'fox']:
            being_animal_vertebrates_mammals_carnivore_canid.append([i])
        elif name in [b'lion', b'tiger', b'leopard']:
            being_animal_vertebrates_mammals_carnivore_felid.append([i])
        elif name in [b'ray', b'shark']:
            being_animal_aquatic_fish_el.append([i])

        else:
            raise ValueError(name + ' did not have a match')

    # Build hierarchy

    terrestrial_artificial_structure.append(terrestrial_artificial_structure_building)
    terrestrial_artificial.append(terrestrial_artificial_structure)
    terrestrial.append(terrestrial_artificial)
    terrestrial.append(terrestrial_natural)

    flora_plant.append(flora_plant_fruit)
    flora_plant.append(flora_plant_flower)
    flora.append(flora_tree)
    flora.append(flora_plant)

    object_vehicle.append(object_vehicle_transport)
    object_vehicle.append(object_vehicle_two)
    object_vehicle.append(object_vehicle_four)
    object_container.append(object_container_food)
    object_container.append(object_container_drink)
    object_furniture.append(object_furniture_seat)
    object.append(object_vehicle)
    object.append(object_container)
    object.append(object_electronic)
    object.append(object_furniture)

    being_person.append(being_person_adult)
    being_person.append(being_person_child)
    being_animal_vertebrates_mammals_carnivore.append(being_animal_vertebrates_mammals_carnivore_felid)
    being_animal_vertebrates_mammals_carnivore.append(being_animal_vertebrates_mammals_carnivore_canid)
    being_animal_vertebrates_mammals.append(being_animal_vertebrates_mammals_marsupial)
    being_animal_vertebrates_mammals.append(being_animal_vertebrates_mammals_carnivore)
    being_animal_vertebrates_mammals.append(being_animal_vertebrates_mammals_rodent)
    being_animal_vertebrates.append(being_animal_vertebrates_reptiles)
    being_animal_vertebrates.append(being_animal_vertebrates_mammals)
    being_animal_aquatic_fish.append(being_animal_aquatic_fish_el)
    being_animal_aquatic.append(being_animal_aquatic_fish)
    being_animal_aquatic.append(being_animal_aquatic_mammal)
    being_animal_invertebrates.append(being_animal_invertebrates_insects)
    being_animal.append(being_animal_vertebrates)
    being_animal.append(being_animal_invertebrates)
    being_animal.append(being_animal_aquatic)
    being.append(being_person)
    being.append(being_animal)

    tree = [terrestrial, sea, cloud, being, flora, object]

    return tree
