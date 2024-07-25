from monai import transforms, data
import numpy as np
import os
import nibabel as nib
def _get_transform():
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["CT_image", "MRI_image","CT_label", "MRI_label"],dtype=np.float32),
            transforms.EnsureChannelFirstd(keys=["CT_image", "MRI_image","CT_label", "MRI_label"]), #selected
            transforms.Resized(keys=["CT_image", "MRI_image","CT_label", "MRI_label"],spatial_size=(96, 96, 96),mode='nearest'), #selected
            transforms.ScaleIntensityRanged(
                keys=["CT_image"], a_min=-1024, a_max=2976,
                b_min=0.0, b_max=1.0, clip=True,
            ), #selected
            transforms.ScaleIntensityRanged(
                keys=["MRI_image"], a_min=0, a_max=1093,
                b_min=0.0, b_max=1.0, clip=True,
            ),  # selected
            transforms.ToTensord(keys=["CT_image", "MRI_image","CT_label", "MRI_label"]),
        ]
        )

    return train_transform

def process(save_root, loader,datalist):
    ind = 0
    for batch_data in loader:
        name_ = datalist[ind]['CT_image'].split('/')[-1]
        name = name_.split('_CT')[0]
        CT_images, CT_labels, MRI_images, MRI_labels = batch_data['CT_image'],batch_data['CT_label'], batch_data['MRI_image'], batch_data['MRI_label']
        CT_image = CT_images[0,0].numpy()
        CT_label = CT_labels[0,0].numpy()
        MRI_image = MRI_images[0,0].numpy()
        MRI_label = MRI_labels[0,0].numpy()

        # save to .npz file
        np.savez(os.path.join(save_root, name), CT_image=CT_image, CT_label=CT_label,MRI_image=MRI_image,MRI_label=MRI_label)
        ind = ind + 1
        print(ind)

def process_data():
    save_root = './dataset/dataset_amos/96_val_npz'
    base_dir = './dataset/dataset_amos/val'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    datalist_json = './dataset/dataset_amos/val_list.json'
    datalist = data.load_decathlon_datalist(datalist_json, True, "validation",
                                            base_dir=base_dir)

    transform = _get_transform()
    ds = data.Dataset(data=datalist, transform=transform)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    process(save_root, loader,datalist)


if __name__ == "__main__":
    process_data()
