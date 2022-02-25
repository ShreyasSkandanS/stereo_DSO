from bench import VkittiDataset, KittiDataset, TartanAirDataset

if __name__ == "__main__":
    vk2 = VkittiDataset()
    sdso_vk2 = vk2.get_sdso(0, False)
    sdso_vk2.run()
    sdso_vk2.save()

    #kit = KittiDataset()

    #tta = TartanAirDataset()
