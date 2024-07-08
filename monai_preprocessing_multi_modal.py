from monai import transforms, data
import numpy as np
import os
import nibabel as nib
def _get_transform():
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["CT_image", "MRI_image","CT_label", "MRI_label"],dtype=np.float32),
            # transforms.AddChannel(),
            transforms.EnsureChannelFirstd(keys=["CT_image", "MRI_image","CT_label", "MRI_label"]), #selected
            transforms.Resized(keys=["CT_image", "MRI_image","CT_label", "MRI_label"],spatial_size=(96, 96, 96),mode='nearest'), #selected
            # transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=["CT_image"], a_min=-1024, a_max=2976,
                b_min=0.0, b_max=1.0, clip=True,
            ), #selected
            transforms.ScaleIntensityRanged(
                keys=["MRI_image"], a_min=0, a_max=1093,
                b_min=0.0, b_max=1.0, clip=True,
            ),  # selected
            # transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
            # transforms.SpatialPadd(keys=["image", "label"], method='symmetric', mode='constant', spatial_size=(96, 96, 96)),
            # transforms.SpatialCropD(keys=["image", "label"], roi_start=(0, 0, 0), roi_end=(96, 96, 96)),
            # transforms.SpatialCropD(keys=["image", "label"], roi_start=(0, 0, 0), roi_end=(96, 96, 96)),
            # transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96)),# selected
            transforms.ToTensord(keys=["CT_image", "MRI_image","CT_label", "MRI_label"]),
        ]
        )

    return train_transform


def find_smallest_box(seg):
    x_start, x_end = np.where(np.any(seg, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(seg, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(seg, axis=(0, 1)))[0][[0, -1]]

    x_start, x_end = max(0, x_start-5), min(seg.shape[0], x_end+5)
    y_start, y_end = max(0, y_start-5), min(seg.shape[1], y_end+5)
    z_start, z_end = max(0, z_start-5), min(seg.shape[2], z_end+5)

    return (x_start, x_end, y_start, y_end, z_start, z_end)

def process(save_root, loader,datalist):
    ind = 0
    for batch_data in loader:
        name_ = datalist[ind]['CT_image'].split('/')[-1]
        name = name_.split('_')[0]
        # inputs = inputs[0].numpy()
        CT_images, CT_labels, MRI_images, MRI_labels = batch_data['CT_image'],batch_data['CT_label'], batch_data['MRI_image'], batch_data['MRI_label']
        CT_image = CT_images[0,0].numpy()
        CT_label = CT_labels[0,0].numpy()
        MRI_image = MRI_images[0,0].numpy()
        MRI_label = MRI_labels[0,0].numpy()

        # name = batch_data["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
        # name = batch_data['filename_or_obj'][0].split('/')[-1]
        # name = name.split('.')[0]

        # save to .npz file
        # np.savez(os.path.join(save_root, name), image=crop_input, seg=crop_seg)
        np.savez(os.path.join(save_root, name), CT_image=CT_image, CT_label=CT_label,MRI_image=MRI_image,MRI_label=MRI_label)
        ind = ind + 1

def process_train():
    # save_root = "/media/zxg/41dcb141-87ae-40f4-88ce-c566824e113b/home/zxg_big_file/RSNA_ADT/monai_process_data/train_npz"
    save_root = "/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/val_npz"

    # img = nib.load('/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/train/raw_img/MRI/0001_MRI.nii.gz')
    # data = img.get_fdata()
    # max_value= np.max(data) # CT: 2976 # label 15 MRI: 1093 label: 13
    # min_value= np.min(data) #CT:-1024 # label 0 MRI: 0 label:0
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    datalist_json = '/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/val_list.json'
    datalist = data.load_decathlon_datalist(datalist_json, True, "training",
                                            base_dir='/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/validation')

    transform = _get_transform()
    ds = data.Dataset(data=datalist, transform=transform)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    process(save_root, loader,datalist)


def process_test():
    save_root = "/media/zxg/41dcb141-87ae-40f4-88ce-c566824e113b/home/zxg_big_file/RSNA_ADT/monai_data/zxg_test_npz"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    datalist_json = '/media/zxg/41dcb141-87ae-40f4-88ce-c566824e113b/home/zxg_big_file/RSNA_ADT/monai_process_data/val_dataset.json'
    datalist = data.load_decathlon_datalist(datalist_json, True, "training", base_dir='/media/zxg/41dcb141-87ae-40f4-88ce-c566824e113b/home/zxg_big_file/RSNA_ADT/monai_process_data/test_npz')
    transform = _get_transform()
    ds = data.Dataset(data=datalist, transform=transform)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    process(save_root, loader,datalist)


if __name__ == "__main__":
    process_train()
    # process_test()
