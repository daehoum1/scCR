import torch
from misc.utils import set_seed, set_filename, setup_logger
from argument import parse_args
import numpy as np
import datetime
import warnings
from scCR import scCR_Trainer

warnings.filterwarnings("ignore")

def main():
    args, _ = parse_args()
    torch.set_num_threads(3)
    
    rmse_list, median_l1_distance_list, cosine_similarity_list = [], [], []
    imputed_ari_list, imputed_nmi_list, imputed_ca_list = [], [], []
 
    file = set_filename(args)
    logger = setup_logger('./', '-', file)
    for seed in range(0, args.n_runs):
        print(f'Seed: {seed}, Filename: {file}')
        set_seed(seed)
        args.seed = seed
        embedder = scCR_Trainer(args)

        if (args.drop_rate != 0.0):
            [rmse, median_l1_distance, cosine_similarity, imputed_ari, imputed_nmi, imputed_ca] = embedder.train()
            
            rmse_list.append(rmse)
            median_l1_distance_list.append(median_l1_distance)
            cosine_similarity_list.append(cosine_similarity)
        
        else:
            [imputed_ari, imputed_nmi, imputed_ca] = embedder.train()
        
        imputed_ari_list.append(imputed_ari)
        imputed_nmi_list.append(imputed_nmi)
        imputed_ca_list.append(imputed_ca)


    logger.info('')
    logger.info(datetime.datetime.now())
    logger.info(file)
    if args.drop_rate > 0.0:
        logger.info(f'-------------------- Drop Rate: {args.drop_rate} --------------------')
        logger.info('[Averaged result]  RMSE  Median_L1  Cosine_Similarity')
        logger.info('{:.3f}+{:.3f} {:.3f}+{:.3f} {:.3f}+{:.3f}'.format(np.round(np.mean(rmse_list),3), np.round(np.std(rmse_list),3), np.round(np.mean(median_l1_distance_list),3), np.round(np.std(median_l1_distance_list),3), np.round(np.mean(cosine_similarity_list),3), np.round(np.std(cosine_similarity_list),3)))
    logger.info('[Averaged result] (Imputed) ARI  NMI  CA')
    logger.info('{:.3f}+{:.3f} {:.3f}+{:.3f} {:.3f}+{:.3f}'.format(np.round(np.mean(imputed_ari_list),3), np.round(np.std(imputed_ari_list),3), np.round(np.mean(imputed_nmi_list),3), np.round(np.std(imputed_nmi_list),3), np.round(np.mean(imputed_ca_list),3), np.round(np.std(imputed_ca_list),3)))
    logger.info('')
    logger.info(args)
    logger.info(f'=================================')

if __name__ == "__main__":
    main()