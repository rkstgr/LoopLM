from names_dataset import NameDataset, NameWrapper

nd = NameDataset()

us_male = nd.get_top_names(n=500, gender="Male", country_alpha2="US")
us_female = nd.get_top_names(n=500, gender="F", country_alpha2="US")

surnames = nd.get_top_names(n=1000, use_first_names=False, country_alpha2="US")

with open("firstnames_1000.txt", "w") as f:
    for name in us_male["US"]["M"]:
        f.write(name)
        f.write("\n")
    for name in us_female["US"]["F"]:
        f.write(name)
        f.write("\n")

with open("surnames_1000.txt", "w") as f:
    for sname in surnames["US"]:
        f.write(sname)
        f.write("\n")
