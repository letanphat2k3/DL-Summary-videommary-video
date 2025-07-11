import argparse
import json
import numpy as np
import torch
from pathlib import Path

from helpers import init_helper, vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Model type: anchor-based or anchor-free")
    parser.add_argument('--ckpt-path', type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument('--source', type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument('--save-json', type=str, required=True, help="Output path for .json label file")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: cuda or cpu")
    parser.add_argument('--nms-thresh', type=float, default=0.5, help="NMS threshold")
    parser.add_argument('--sample-rate', type=int, default=15, help="Video sampling rate")

    # Required for get_model
    parser.add_argument('--base-model', type=str, default='attention')
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--anchor-scales', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--num-head', type=int, default=8)

    args = parser.parse_args()

    # Load model
    print('Loading model...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    # Load weights
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # Extract features from video
    print('Processing video...')
    video_proc = video_helper.VideoPreprocessor(args.sample_rate)
    n_frames, seq, cps, nfps, picks = video_proc.run(args.source)
    seq_len = len(seq)

    # Predict summary
    print('Predicting...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)
        pred_cls, pred_bboxes = model.predict(seq_torch)
        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)
        pred_summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

    # Convert to JSON format
    user_summary = [pred_summ.astype(int).tolist()]  # single "annotator" is the model
    output_data = {"user_summary": user_summary}

    # Save to file
    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved JSON label to: {args.save_json}")


if __name__ == '__main__':
    main()
