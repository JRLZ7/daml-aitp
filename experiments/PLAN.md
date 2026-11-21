# Experiment plan

We have validated data and a fixed split. **No model has been trained or scored
yet.** All improvements below are hypotheses, not reported achievements.

## Where we can earn an advantage

Our proposed focus is to preserve useful room context, identify which visual
evidence models miss, and spend compute on changes that fix those errors.
AI can help generate code; the team must choose questions and assess evidence.

1. **Make comparisons trustworthy.** Use the same split and inference transforms
   for controlled comparisons. Save exact configs and individual predictions.
2. **Study validation failures.** After the baseline, inspect a reproducible sample
   of correct/incorrect validation examples. Group errors by confusion pair,
   lost room context, small-object detail, lighting, and ambiguous categories.
   Save notes; do not infer failure patterns from anecdotes alone. Do not manually
   relabel data or use test-image inspection to recover answers.
3. **Test whole-room and object evidence.** Begin with gentler crops or a
   whole-image view, then resolution. If justified, compare a simple classifier
   on concatenated global and pooled local ImageNet features. Keep the encoder
   and data source fixed so we can attribute the change to views/pooling.
4. **Choose ensemble partners by measured benefit.** Compare each model's accuracy,
   prediction disagreement, and corrections/regressions. Start with equal-weight
   probability averaging; align class order. Disagreement alone is not a gain.
5. **Confirm finalists.** Reuse frozen choices on an additional predeclared split
   or a small cross-validation comparison using provided labeled data only.
   A different training seed tests optimizer variability; a different split
   also tests sensitivity to validation examples. Record both separately.

## Initial experiment queue

Run IDs are reserved, not completed. Fix exact configs and checkpoint sources
before running. Keep the baseline, architecture, and preprocessing comparisons
separate where possible.

| ID | Question | Main change | Evidence to retain |
| --- | --- | --- | --- |
| E001 | How useful are frozen ImageNet features? | ResNet-50 + trained linear head | Accuracy, timing, predictions, source/checksum |
| E002 | Does adapting the encoder help? | Fine-tune E001 encoder | Same split; compare with E001 |
| E003 | Does another backbone help? | ImageNet ConvNeXt-Tiny, matched input policy | Compare with E002; log training recipe differences |
| E004 | Are crops removing needed scene context? | Change crop/context policy on selected model | Accuracy plus per-class and paired-error differences |
| E005 | Is fine detail limiting recognition? | Increase resolution on same model | Accuracy gain versus memory/runtime |
| E006 | Do global and local views complement each other? | Whole-image + pooled local features | Ablate global-only/local-only/combined; cost |
| E007 | Do two trained models complement each other? | Equal-weight probability ensemble | Accuracy, corrections, regressions, inference cost |
| E008 | Do finalist gains persist? | Predeclared confirmation split/seeds | Same frozen choices on additional comparisons |

E006 is optional: proceed only if error analysis and compute justify it. When
testing changes together, describe the result as a combined recipe rather than
attributing all gains to one component. No target accuracy is promised.

## Evidence and stopping rules

- Add a dated decision entry before a consequential choice and link its run IDs.
  Before each run state the hypothesis, reference run, changed factors, and how
  the outcome would affect the next choice.
- Record all completed runs, including negative results; retain failed-run notes
  with a failure reason. Do not silently replace a run's configuration or metrics.
- One extra correct image is about 0.094 percentage points on 1,066 examples.
  A few examples' improvement is a candidate for confirmation, not decisive proof.
  Paired resampling can help describe uncertainty but does not remove bias from
  repeated tuning on the same validation set.
- Repeatedly inspect validation results with care. An additional split drawn from
  these same labeled records is confirmation, not a fresh independent test set.
- If context/resolution changes do not give repeatable gains, return to the best
  simple recipe. Do not build a complex feature pipeline just to be different.
- Keep test data out of fitting, augmentation tuning, label review, and feature
  normalization statistics. Use it only for automated input checks and final
  inference. Public leaderboard performance is a secondary check.
- Retain a runnable single-model solution before experimenting with ensembles.
  Reserve time before the deadline for final inference and package verification.

## Reading and source provenance

These are methodological references, not permission to reuse scene-trained
weights or published answers. Use the provided labeled data for all learning;
record the exact training data and identifier of every pretrained checkpoint.

- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): frozen features versus encoder fine-tuning.
- [Torchvision models and weights](https://docs.pytorch.org/vision/stable/models.html): candidate ImageNet checkpoint metadata and transforms.
- [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html): ConvNeXt architecture reference.
- [Multi-scale Orderless Pooling of Deep Convolutional Activation Features](https://arxiv.org/abs/1403.1840): motivation for testing complementary global/local representations. Our proposed simple pooling experiment is not a reproduction of that method.
