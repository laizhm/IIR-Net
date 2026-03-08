# used_service = '509'
used_service = '3090'
gpu_id = 1#########################todo gzl

num_epochs = 30
GRL_epochs = 20

imageSize = 256
maskSize = 256

num_workers = 8 # todo gzl orign:4
num_workers2 = 0

print_freq = 2000

sample_size = 2
num_domains = 4
batch_size = sample_size  * 32 * num_domains # todo gzl orign:4
split_ratio = 2     # Ds:Dt=(split_ratio):(num_domains-split_ratio)

train_num_samples = 10000
valid_num_samples = 2000
test_num_samples = 3000

optim_name = 'Adam'
model_Adam_lr = 1e-4
model_SGD_lr = 0.01


"""输出结果，TXT文件"""
# result_file_root = './results/Illumination_FF++(real_4subs)(4fakes)(gap5)(faceRotate_F)(Aug_T)(decoder_seg)(new)(Dis_2FC2)(adv_anchor)'
result_file_root = './results/Illumination_FF++(real_4subs)(4fakes)(gap5)(faceRotate_F)(Aug_T)'############todo gzl

result_valid_file = result_file_root + '/result_valid_file.txt'
result_test_file = result_file_root + '/result_test_file.txt'
"""预测mask文件"""
result_figures_path = result_file_root + '/figures'

"""模型保存文件夹"""
# saveRoot_models = './checkpoints/Illumination_FF++(real_4subs)(4fakes)(gap5)(faceRotate_F)(Aug_T)(decoder_seg)(new)(Dis_2FC2)(adv_anchor)'
saveRoot_models = './checkpoints/Illumination_FF++(real_4subs)(4fakes)(gap5)(faceRotate_F)(Aug_T)'##############todo gzl
saveRoot_Illumination = saveRoot_models + '/Illumination_6_7/'
# saveRoot_Illumination = saveRoot_models + '/Illumination_6_7/sbi/'
# saveRoot_Illumination = saveRoot_models + '/Illumination_6_7/spsl/'

"""模型加载文件"""
# savePath_Illumination = saveRoot_Illumination + 'old/c23/' + 'Illumination_epoch14.pth'##todo gzl
# savePath_Illumination = saveRoot_Illumination + 'Illumination_epoch11.pth'##todo gzl
savePath_Illumination = saveRoot_Illumination + 'Illumination_epoch.pth'##todo gzl
# savePath_Illumination = saveRoot_Illumination + 'Illumination_epoch16.pth'##todo gzl

# savePath_Relighting = './checkpoints/model_relighting/model_epoch99.pth'            # 实例1的Relighting模型
savePath_Relighting = './checkpoints/model_lighting_transfer/model_epoch106.pth'    # 实例2的Relighting模型（使用中）

"""训练、验证和测试数据"""
# train_dataPath_D1_T = ''
# train_dataPath_D1_F = ''
# train_dataPath_D2_T = ''
# train_dataPath_D2_F = ''
# train_dataPath_D3_T = ''
# train_dataPath_D3_F = ''
# train_dataPath_D4_T = ''
# train_dataPath_D4_F = ''
#
# valid_dataPath_T = ''
# valid_dataPath_F = ''
# test_dataPath_T = ''
# test_dataPath_F = ''
# test_dataPath_S = ''


"""数据集文件"""
"""FF++库内不同伪造方法的数据集"""
# 1.1 所有数据集
##########################正样本##############################
FF_dataPath_real_c0_train = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_train(face).txt'
FF_dataPath_real_c23_train = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_train(face).txt'
FF_dataPath_real_c40_train = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_train(face).txt'
FF_dataPath_real_c0_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_devel(face).txt'
FF_dataPath_real_c23_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_devel(face).txt'
FF_dataPath_real_c40_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_devel(face).txt'
FF_dataPath_real_c0_test = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_test(face).txt'
FF_dataPath_real_c23_test = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_test(face).txt'
FF_dataPath_real_c40_test = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_test(face).txt'
#
FF_dataPath_real_c0_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_train(face)(gap5).txt'
FF_dataPath_real_c23_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_train(face)(gap5).txt'
FF_dataPath_real_c40_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_train(face)(gap5).txt'
FF_dataPath_real_c0_train_gap5_sub1 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_train(face)(gap5)(sub1).txt'
FF_dataPath_real_c0_train_gap5_sub2 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_train(face)(gap5)(sub2).txt'
FF_dataPath_real_c0_train_gap5_sub3 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_train(face)(gap5)(sub3).txt'
FF_dataPath_real_c0_train_gap5_sub4 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c0_train(face)(gap5)(sub4).txt'
FF_dataPath_real_c23_train_gap5_sub1 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_train(face)(gap5)(sub1).txt'
FF_dataPath_real_c23_train_gap5_sub2 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_train(face)(gap5)(sub2).txt'
FF_dataPath_real_c23_train_gap5_sub3 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_train(face)(gap5)(sub3).txt'
FF_dataPath_real_c23_train_gap5_sub4 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c23_train(face)(gap5)(sub4).txt'
FF_dataPath_real_c40_train_gap5_sub1 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_train(face)(gap5)(sub1).txt'
FF_dataPath_real_c40_train_gap5_sub2 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_train(face)(gap5)(sub2).txt'
FF_dataPath_real_c40_train_gap5_sub3 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_train(face)(gap5)(sub3).txt'
FF_dataPath_real_c40_train_gap5_sub4 = './your_path/data_FF++/3090/FaceForensics_dataPath_real_c40_train(face)(gap5)(sub4).txt'
##########################负样本##############################
# 1.3 FF++ DF
FF_dataPath_DF_c0_train = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c0_train(face).txt'
FF_dataPath_DF_c23_train = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c23_train(face).txt'
FF_dataPath_DF_c40_train = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c40_train(face).txt'
FF_dataPath_DF_c0_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c0_train(face)(gap5).txt'
FF_dataPath_DF_c23_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c23_train(face)(gap5).txt'
FF_dataPath_DF_c40_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c40_train(face)(gap5).txt'
FF_dataPath_DF_c0_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c0_devel(face).txt'
FF_dataPath_DF_c23_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c23_devel(face).txt'
FF_dataPath_DF_c40_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c40_devel(face).txt'
FF_dataPath_DF_c0_test = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c0_test(face).txt'
FF_dataPath_DF_c23_test = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c23_test(face).txt'
FF_dataPath_DF_c40_test = './your_path/data_FF++/3090/FaceForensics_dataPath_Deepfakes_c40_test(face).txt'
# 1.4 FF++ FS
FF_dataPath_FS_c0_train = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c0_train(face).txt'
FF_dataPath_FS_c23_train = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c23_train(face).txt'
FF_dataPath_FS_c40_train = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c40_train(face).txt'
FF_dataPath_FS_c0_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c0_train(face)(gap5).txt'
FF_dataPath_FS_c23_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c23_train(face)(gap5).txt'
FF_dataPath_FS_c40_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c40_train(face)(gap5).txt'
FF_dataPath_FS_c0_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c0_devel(face).txt'
FF_dataPath_FS_c23_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c23_devel(face).txt'
FF_dataPath_FS_c40_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c40_devel(face).txt'
FF_dataPath_FS_c0_test = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c0_test(face).txt'
FF_dataPath_FS_c23_test = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c23_test(face).txt'
FF_dataPath_FS_c40_test = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceSwap_c40_test(face).txt'
# 1.5 FF++ F2F
FF_dataPath_F2F_c0_train = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c0_train(face).txt'
FF_dataPath_F2F_c23_train = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c23_train(face).txt'
FF_dataPath_F2F_c40_train = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c40_train(face).txt'
FF_dataPath_F2F_c0_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c0_train(face)(gap5).txt'
FF_dataPath_F2F_c23_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c23_train(face)(gap5).txt'
FF_dataPath_F2F_c40_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c40_train(face)(gap5).txt'
FF_dataPath_F2F_c0_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c0_devel(face).txt'
FF_dataPath_F2F_c23_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c23_devel(face).txt'
FF_dataPath_F2F_c40_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c40_devel(face).txt'
FF_dataPath_F2F_c0_test = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c0_test(face).txt'
FF_dataPath_F2F_c23_test = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c23_test(face).txt'
FF_dataPath_F2F_c40_test = './your_path/data_FF++/3090/FaceForensics_dataPath_Face2Face_c40_test(face).txt'
# 1.6 FF++ NT
FF_dataPath_NT_c0_train = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c0_train(face).txt'
FF_dataPath_NT_c23_train = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c23_train(face).txt'
FF_dataPath_NT_c40_train = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c40_train(face).txt'
FF_dataPath_NT_c0_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c0_train(face)(gap5).txt'
FF_dataPath_NT_c23_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c23_train(face)(gap5).txt'
FF_dataPath_NT_c40_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c40_train(face)(gap5).txt'
FF_dataPath_NT_c0_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c0_devel(face).txt'
FF_dataPath_NT_c23_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c23_devel(face).txt'
FF_dataPath_NT_c40_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c40_devel(face).txt'
FF_dataPath_NT_c0_test = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c0_test(face).txt'
FF_dataPath_NT_c23_test = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c23_test(face).txt'
FF_dataPath_NT_c40_test = './your_path/data_FF++/3090/FaceForensics_dataPath_NeuralTextures_c40_test(face).txt'
# 1.7 FF++ FSh
FF_dataPath_FSh_c0_train = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c0_train(face).txt'
FF_dataPath_FSh_c23_train = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c23_train(face).txt'
FF_dataPath_FSh_c40_train = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c40_train(face).txt'
FF_dataPath_FSh_c0_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c0_train(face)_gap5.txt'
FF_dataPath_FSh_c23_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c23_train(face)_gap5.txt'
FF_dataPath_FSh_c40_train_gap5 = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c40_train(face)_gap5.txt'
FF_dataPath_FSh_c0_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c0_devel(face).txt'
FF_dataPath_FSh_c23_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c23_devel(face).txt'
FF_dataPath_FSh_c40_valid = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c40_devel(face).txt'
FF_dataPath_FSh_c0_test = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c0_test(face).txt'
FF_dataPath_FSh_c23_test = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c23_test(face).txt'
FF_dataPath_FSh_c40_test = './your_path/data_FF++/3090/FaceForensics_dataPath_FaceShifter_c40_test(face).txt'

"""TIMIT库内不同伪造方法的数据集，正负样本合并在一起了"""
TIMIT_dataPath_train = './your_path/data_TIMIT/TIMIT_Input_train_path.txt'
TIMIT_dataPath_valid = './your_path/data_TIMIT/TIMIT_Input_validation_path.txt'
TIMIT_dataPath_test = './your_path/data_TIMIT/TIMIT_Input_test_path.txt'
TIMIT_dataPath_test_HQ = './your_path/data_TIMIT/TIMIT_Input_test_path_H.txt'
TIMIT_dataPath_test_LQ = './your_path/data_TIMIT/TIMIT_Input_test_path_L.txt'

"""DFD库内不同伪造方法的数据集，正负样本合并在一起"""
# DFD数据集为旧的数据，未重新检测制作
DFD_dataPath_c23_train = './your_path/data_DFD/DeepFakeDetection_Input_train_path(c23).txt'
DFD_dataPath_c23_train_10w = './your_path/data_DFD/DeepFakeDetection_Input_train_path(c23)_10w.txt'
DFD_dataPath_c40_train = './your_path/data_DFD/DeepFakeDetection_Input_train_path(c40).txt'
DFD_dataPath_c23_valid = './your_path/data_DFD/DeepFakeDetection_Input_validation_path(c23).txt'
DFD_dataPath_c40_valid = './your_path/data_DFD/DeepFakeDetection_Input_validation_path(c40).txt'
DFD_dataPath_c23_test = './your_path/data_DFD/DeepFakeDetection_Input_test_path(c23).txt'
DFD_dataPath_c40_test = './your_path/data_DFD/DeepFakeDetection_Input_test_path(c40).txt'

"""DFDC-P库内不同伪造方法的数据集"""
DFDC_P_dataPath_real_ori_test = './your_path/data_DFDC-P/DFDC-P_Input_test_real(original)_path(face).txt'
DFDC_P_dataPath_real_low_quality_test = './your_path/data_DFDC-P/DFDC-P_Input_test_real(low_quality)_path(face).txt'
DFDC_P_dataPath_real_low_fps_test = './your_path/data_DFDC-P/DFDC-P_Input_test_real(low_fps)_path(face).txt'
DFDC_P_dataPath_real_low_res_test = './your_path/data_DFDC-P/DFDC-P_Input_test_real(low_res)_path(face).txt'

DFDC_P_dataPath_fake_ori_test = './your_path/data_DFDC-P/DFDC-P_Input_test_fake(original)_path(face).txt'
DFDC_P_dataPath_fake_low_quality_test = './your_path/data_DFDC-P/DFDC-P_Input_test_fake(low_quality)_path(face).txt'
DFDC_P_dataPath_fake_low_fps_test = './your_path/data_DFDC-P/DFDC-P_Input_test_fake(low_fps)_path(face).txt'
DFDC_P_dataPath_fake_low_res_test = './your_path/data_DFDC-P/DFDC-P_Input_test_fake(low_res)_path(face).txt'

"""DFDC库内不同伪造方法的数据集"""
DFDC_dataPath_real_train = './your_path/data_DFDC/DFDC_Input_train_real_path(3090)(sample_gap_5)(face).txt'
DFDC_dataPath_real_test = './your_path/data_DFDC/DFDC_Input_test_real_path(3090)(face).txt'

DFDC_dataPath_fake_train = './your_path/data_DFDC/DFDC_Input_train_fake_path(3090)(sample_gap_5)(face).txt'
DFDC_dataPath_fake_test = './your_path/data_DFDC/DFDC_Input_test_fake_path(3090)(face).txt'

"""CDF_v1库内不同伪造方法的数据集，正负样本合并在一起"""
CDF_v1_dataPath_train = './your_path/data_CDF/Celeb_DF_Input_train_path.txt'
CDF_v1_dataPath_train_10w = './your_path/data_CDF/Celeb_DF_Input_train_path_10w.txt'
CDF_v1_dataPath_test = './your_path/data_CDF/Celeb_DF_Input_test_path.txt'

"""CDF_v2库内不同伪造方法的数据集"""
CDF_v2_dataPath_real_test = './your_path/data_CDF/3090/Celeb_DF_v2_Input_test_real_path(face).txt'
CDF_v2_dataPath_fake_test = './your_path/data_CDF/3090/Celeb_DF_v2_Input_test_fake_path(face).txt'

"""Deeper库内不同伪造方法的数据集，正负样本合并在一起"""
# 最新跑的数据集
Deeper_dataPath_test_new = './your_path/data_Deeper/DeeperForensics_path_test(sample_gap_10)(face).txt'
# 旧的数据集
Deeper_dataPath_train_old = './your_path/data_Deeper/DeeperForensics_path_train(sample_gap_5)(face).txt'
Deeper_dataPath_valid_old = './your_path/data_Deeper/DeeperForensics_path_validation(sample_gap_5)(face).txt'
Deeper_dataPath_test_old = './your_path/data_Deeper/DeeperForensics_path_test(sample_gap_5)(face).txt'
