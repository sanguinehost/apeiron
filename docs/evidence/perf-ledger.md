# Performance ledger (rolled up per-run receipts)

Rolled up from: `evidence/perf-global-baseline.log`, `evidence/perf-global-baseline.md`, `evidence/perf-global-baseline.stderr.log`, `evidence/perf-micro-spatial-2026-09-19.md`, `evidence/perf-remeasure-2026-09-19-post-correctness.md`, `evidence/perf-remeasure-2026-09-19-post-mask.md`, `evidence/perf-remeasure-2026-09-19-post-scatter.md`, `evidence/perf-remeasure-2026-09-19-post-spill.md`, `evidence/perf-remeasure-2026-09-19.md`, `evidence/perf-rework-first-cut-baseline.md`.

apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes
hashtrinity_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358
rustc=rustc 1.97.0-nightly (17584a181 2026-04-13)
cargo=cargo 1.97.0-nightly (eb94155a9 2026-04-09)
target=x86_64-linux profile=release
workload=reg1-performance-full-shared-fixed16
cpu=Intel(R) Core(TM) i9-14900KF
kernel=7.2.3-1-cachyos
threads=1 single_thread=true
cache_condition=warmed by 5 warmup traces; tables rebuilt and touched before every sample
apeiron_table_bytes=40 candidate_table_bytes=133128 hashtrinity_capacity=8192 scheduled_ops=2097152 modes=2
home_boundary_keys=4096 full_word_ok=true residue_ok=true
collision_keys=2 word_equal=true occupancy=2 capacity=4 first_ok=true second_ok=true zero_refused=true zero_default=true
profile=full lookups=1048576 updates=1048576 warmup=5 samples=31 modes=2
full_profile lookups=1048576 updates=1048576 warmup=5 samples=31
mode=replacement apeiron_min_ns=7136135 apeiron_median_ns=7166043 apeiron_max_ns=7662095 candidate_min_ns=7536569 candidate_median_ns=7676069 candidate_max_ns=8121410 ht_min_ns=2960256 ht_median_ns=3012893 ht_max_ns=3487527 ratio=2.3785 candidate_ht_ratio=2.5477 candidate_current_ratio=1.0712 apeiron_ns_per_op=3.417 candidate_ns_per_op=3.660 ht_ns_per_op=1.437
mode=replacement apeiron_samples_ns=7173214,7176696,7190261,7153505,7177964,7140646,7165312,7146995,7485873,7143243,7505764,7149475,7662095,7578235,7168769,7518461,7166927,7145978,7170541,7150508,7152324,7159297,7136135,7166043,7146064,7148636,7160305,7208667,7372697,7137949,7617010
mode=replacement candidate_samples_ns=7651232,8121410,8083207,7723310,8037900,7718827,7613409,7698912,7536569,7645923,7617530,7720146,7603433,7545033,7547229,7585993,7663460,7741442,7681425,8095633,7611756,7963656,7626372,8010612,7933396,7577769,7958282,7827340,7671834,7676069,7546013
mode=replacement ht_samples_ns=2960256,3028639,2990587,3012454,3025343,3007148,3487527,2996180,3005354,3038985,3017685,2977703,3005502,3002280,3029729,3083221,3051200,3464761,3036118,3005729,3027261,3014024,2965690,2991106,3039434,2983249,2980087,3012893,3408950,2994443,3032963
mode=wrapping-add apeiron_min_ns=7136754 apeiron_median_ns=7162275 apeiron_max_ns=7626448 candidate_min_ns=7630636 candidate_median_ns=7758551 candidate_max_ns=8219071 ht_min_ns=2983480 ht_median_ns=3029795 ht_max_ns=3475480 ratio=2.3639 candidate_ht_ratio=2.5608 candidate_current_ratio=1.0833 apeiron_ns_per_op=3.415 candidate_ns_per_op=3.700 ht_ns_per_op=1.445
mode=wrapping-add apeiron_samples_ns=7141747,7145967,7177799,7154965,7153889,7168922,7164940,7181720,7144978,7231000,7136754,7162593,7582980,7172663,7626448,7164151,7577635,7590284,7149731,7602948,7154108,7149360,7146521,7164567,7152032,7152422,7162275,7150760,7140472,7142696,7213784
mode=wrapping-add candidate_samples_ns=7701308,7808302,7692644,8218347,7736973,8134502,8166265,7739025,8119019,7770850,7697456,7701857,7773183,7726019,7630636,7760981,7675497,7758551,7843075,7703503,7692337,7712297,7707690,8152483,7763261,8219071,8053920,7693233,8088924,7698443,8123926
mode=wrapping-add ht_samples_ns=3013566,3406165,3044892,3059511,3061908,3012462,3002216,3011608,3022466,3032280,3475480,3142080,3039057,3015667,3009957,3027400,3031511,3027350,3037823,2983480,3014677,3439869,3029795,3034735,3050099,3009728,3030491,3082915,3025204,3024359,2989631
full_projection_seconds=0.73 reference_seconds=73 reproduces_reference=false note="projected from smoke medians, not a measurement; band +/-25%"
strata_protocol profile=full apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes hashtrinity_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc="rustc 1.97.0-nightly (17584a181 2026-04-13)" cpu="Intel(R) Core(TM) i9-14900KF" campaigns=3 warmup=2 samples=15 ops=4096 seed=0x5eed1234abcd0001 oracle_timing=before_timing note="every strata cell line shares this protocol"
strata cell=occ1-direct-index-resident-present-hit-replacement occupancy=1 distribution=direct-index-resident operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=d5c179af3fc2f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12276 ht_median_ns=2906 ratio=4.2244 campaign_ratios=[4.2268,4.2266,4.2226] ratio_interval=[4.2226,4.2268] apeiron_samples_interval=[12254,12318] ht_samples_interval=[2898,3399] apeiron_ns_per_op=2.9971 ht_ns_per_op=0.7095 status=fail reason="ratio 4.2244 > 1.00"
strata cell=occ1-direct-index-resident-present-hit-replacement raw_apeiron_samples_ns=12318,12318,12275,12275,12260,12258,12283,12278,12278,12296,12292,12287,12293,12289,12279,12261,12259,12262,12261,12274,12274,12288,12286,12291,12290,12276,12282,12261,12261,12259,12261,12268,12276,12271,12286,12286,12286,12291,12279,12284,12261,12265,12254,12257,12265
strata cell=occ1-direct-index-resident-present-hit-replacement raw_ht_samples_ns=2910,2903,2913,2901,2910,2903,2905,2906,2902,2911,2904,2906,2907,2904,2913,2907,2901,2903,2906,2898,2911,2902,3399,2906,2904,2910,2900,2908,2899,2903,2905,2909,2901,2909,2906,2908,2909,2903,2912,2899,2909,2900,2907,2904,2902
strata cell=occ1-direct-index-resident-present-hit-wrapping-add occupancy=1 distribution=direct-index-resident operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=d5c179af3fc2f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12278 ht_median_ns=2904 ratio=4.2280 campaign_ratios=[4.2287,4.2283,4.2280] ratio_interval=[4.2280,4.2287] apeiron_samples_interval=[12255,14308] ht_samples_interval=[2897,2913] apeiron_ns_per_op=2.9976 ht_ns_per_op=0.7090 status=fail reason="ratio 4.2280 > 1.00"
strata cell=occ1-direct-index-resident-present-hit-wrapping-add raw_apeiron_samples_ns=12280,12281,12262,12269,12261,12261,12266,12265,12285,12280,12297,12291,12296,14053,12292,12279,12283,12259,12262,12258,12260,12264,12262,12279,12273,12288,12284,12293,12291,12288,12270,12274,12256,12260,12258,12255,12266,12261,12278,12275,12290,12291,12288,14308,12294
strata cell=occ1-direct-index-resident-present-hit-wrapping-add raw_ht_samples_ns=2913,2900,2909,2901,2904,2903,2904,2910,2898,2911,2902,2906,2905,2904,2910,2911,2900,2907,2902,2902,2905,2901,2908,2897,2911,2900,2908,2901,2904,2906,2911,2900,2907,2902,2902,2905,2901,2908,2897,2910,2902,2906,2903,2903,2908
strata cell=occ1-offset-beyond-direct-present-hit-replacement occupancy=1 distribution=offset-beyond-direct operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=92b1413489c2f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=22600 ht_median_ns=2918 ratio=7.7450 campaign_ratios=[7.7432,7.7450,7.7676] ratio_interval=[7.7432,7.7676] apeiron_samples_interval=[22529,22724] ht_samples_interval=[2908,4409] apeiron_ns_per_op=5.5176 ht_ns_per_op=0.7124 status=fail reason="ratio 7.7450 > 1.00"
strata cell=occ1-offset-beyond-direct-present-hit-replacement raw_apeiron_samples_ns=22612,22529,22583,22547,22593,22610,22588,22607,22561,22566,22630,22587,22530,22598,22565,22581,22600,22624,22600,22604,22559,22616,22575,22625,22574,22575,22577,22610,22574,22603,22672,22612,22620,22618,22573,22645,22569,22689,22595,22724,22691,22692,22688,22683,22666
strata cell=occ1-offset-beyond-direct-present-hit-replacement raw_ht_samples_ns=2917,2916,2918,2924,2926,2915,2933,2910,2915,2913,2915,2918,2926,2917,2923,4409,2915,2918,2913,2916,2914,2919,2919,2920,2919,2930,2908,2923,2912,2917,2914,2914,2915,2928,2918,2924,2927,2937,2923,2910,2916,2915,2910,2920,2920
strata cell=occ1-offset-beyond-direct-present-hit-wrapping-add occupancy=1 distribution=offset-beyond-direct operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=92b1413489c2f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=22733 ht_median_ns=2917 ratio=7.7933 campaign_ratios=[7.7735,7.7890,7.8076] ratio_interval=[7.7735,7.8076] apeiron_samples_interval=[22547,24526] ht_samples_interval=[2904,2930] apeiron_ns_per_op=5.5500 ht_ns_per_op=0.7122 status=fail reason="ratio 7.7933 > 1.00"
strata cell=occ1-offset-beyond-direct-present-hit-wrapping-add raw_apeiron_samples_ns=22693,22683,22652,22734,22634,22727,22654,22700,22682,22715,24526,22634,22547,22700,22603,22740,22748,22721,22785,22732,22673,22718,22663,22693,22671,22756,22736,22784,22740,22782,22759,22809,22770,22794,22808,22733,22758,22766,22729,22778,22767,22767,22768,22784,22704
strata cell=occ1-offset-beyond-direct-present-hit-wrapping-add raw_ht_samples_ns=2916,2919,2922,2913,2921,2904,2919,2912,2918,2916,2930,2916,2924,2910,2918,2919,2907,2918,2922,2911,2919,2922,2914,2922,2909,2919,2910,2911,2920,2919,2913,2916,2921,2916,2924,2911,2924,2905,2917,2912,2916,2917,2921,2911,2921
strata cell=occ1-seeded-random-present-hit-replacement occupancy=1 distribution=seeded-random operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=2fb06e73fd5bd425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=24064 ht_median_ns=4573 ratio=5.2622 campaign_ratios=[5.2474,5.3269,5.2501] ratio_interval=[5.2474,5.3269] apeiron_samples_interval=[23815,25973] ht_samples_interval=[4356,4677] apeiron_ns_per_op=5.8750 ht_ns_per_op=1.1165 status=fail reason="ratio 5.2622 > 1.00"
strata cell=occ1-seeded-random-present-hit-replacement raw_apeiron_samples_ns=24131,24021,24092,24118,24083,24033,24084,24064,24027,24000,24016,23992,24057,24030,24032,25336,24182,24122,24193,24184,24218,24148,25973,24327,24273,24084,24090,24075,24189,24105,23815,23899,23937,23878,23848,23898,23909,23912,23899,23943,23910,23903,24069,24085,24010
strata cell=occ1-seeded-random-present-hit-replacement raw_ht_samples_ns=4577,4368,4635,4502,4630,4576,4625,4609,4574,4544,4580,4544,4597,4640,4608,4603,4477,4477,4394,4386,4389,4602,4617,4436,4585,4540,4368,4607,4573,4580,4356,4575,4599,4591,4399,4671,4509,4565,4431,4376,4527,4556,4554,4374,4677
strata cell=occ1-seeded-random-present-hit-wrapping-add occupancy=1 distribution=seeded-random operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=2fb06e73fd5bd425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=23981 ht_median_ns=4533 ratio=5.2903 campaign_ratios=[5.2591,5.3049,5.3327] ratio_interval=[5.2591,5.3327] apeiron_samples_interval=[23855,25786] ht_samples_interval=[4336,4625] apeiron_ns_per_op=5.8547 ht_ns_per_op=1.1067 status=fail reason="ratio 5.2903 > 1.00"
strata cell=occ1-seeded-random-present-hit-wrapping-add raw_apeiron_samples_ns=23901,23992,23855,23932,23857,23913,23892,23936,23896,23956,23912,23946,23897,23971,25523,24181,23976,24007,24047,24109,24032,24201,24028,24044,24080,24091,24035,24068,23953,24064,24202,25786,23978,23996,23992,23988,23955,23988,23954,23976,24002,23964,23981,23978,23981
strata cell=occ1-seeded-random-present-hit-wrapping-add raw_ht_samples_ns=4419,4558,4587,4386,4390,4513,4590,4368,4392,4558,4576,4547,4557,4561,4421,4385,4620,4574,4575,4625,4606,4533,4390,4559,4379,4605,4353,4360,4373,4388,4566,4549,4408,4577,4573,4537,4497,4391,4609,4376,4373,4360,4570,4336,4359
strata cell=occ1-clustered-skewed-present-hit-replacement occupancy=1 distribution=clustered-skewed operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=8ba982463c10e225 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=24117 ht_median_ns=2916 ratio=8.2706 campaign_ratios=[8.2650,8.2572,8.2935] ratio_interval=[8.2572,8.2935] apeiron_samples_interval=[23855,25831] ht_samples_interval=[2905,3085] apeiron_ns_per_op=5.8879 ht_ns_per_op=0.7119 status=fail reason="ratio 8.2706 > 1.00"
strata cell=occ1-clustered-skewed-present-hit-replacement raw_apeiron_samples_ns=23995,24263,24003,24206,24274,24142,24256,24593,24106,24429,24038,24354,24116,24029,24092,24320,24117,24067,24120,24009,24078,24026,24008,24067,25831,23964,24109,23935,24292,24160,24100,24332,24207,24224,24098,24258,24135,24166,23855,24231,24024,24096,24159,24069,24238
strata cell=occ1-clustered-skewed-present-hit-replacement raw_ht_samples_ns=2912,2916,2914,2914,2931,2951,2976,2908,2928,2913,2913,2925,3049,3085,2921,2910,2912,2910,2921,2918,2916,2921,2907,2910,2910,2924,2920,2917,2916,2939,2906,2925,2913,2915,2921,2909,2921,2913,2905,2912,2911,2918,2911,2954,2912
strata cell=occ1-clustered-skewed-present-hit-wrapping-add occupancy=1 distribution=clustered-skewed operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=8ba982463c10e225 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=24283 ht_median_ns=2914 ratio=8.3332 campaign_ratios=[8.3393,8.3093,8.3363] ratio_interval=[8.3093,8.3393] apeiron_samples_interval=[23949,26002] ht_samples_interval=[2904,2928] apeiron_ns_per_op=5.9285 ht_ns_per_op=0.7114 status=fail reason="ratio 8.3332 > 1.00"
strata cell=occ1-clustered-skewed-present-hit-wrapping-add raw_apeiron_samples_ns=24308,24284,24515,24386,24214,24384,24148,24400,24287,24264,24196,24339,24093,24077,24056,24299,24372,24151,24252,24053,24158,24287,24230,24011,24503,24359,24328,24153,24160,23949,24315,24283,24029,24385,24185,24474,24037,26002,24173,24317,24292,24491,24097,24293,24209
strata cell=occ1-clustered-skewed-present-hit-wrapping-add raw_ht_samples_ns=2904,2914,2907,2917,2912,2911,2921,2905,2917,2911,2916,2911,2910,2916,2918,2910,2916,2928,2924,2910,2914,2918,2922,2919,2909,2920,2913,2910,2918,2909,2906,2915,2912,2918,2915,2910,2914,2905,2919,2913,2908,2926,2914,2915,2916
strata cell=occ2-direct-index-resident-present-hit-replacement occupancy=2 distribution=direct-index-resident operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=a436221b7910b725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12281 ht_median_ns=2907 ratio=4.2246 campaign_ratios=[4.2229,4.2222,4.2245] ratio_interval=[4.2222,4.2245] apeiron_samples_interval=[12258,12339] ht_samples_interval=[2900,2970] apeiron_ns_per_op=2.9983 ht_ns_per_op=0.7097 status=fail reason="ratio 4.2246 > 1.00"
strata cell=occ2-direct-index-resident-present-hit-replacement raw_apeiron_samples_ns=12334,12310,12296,12291,12307,12281,12312,12270,12272,12276,12266,12263,12263,12272,12265,12298,12292,12339,12290,12292,12281,12292,12269,12274,12262,12272,12258,12265,12268,12261,12285,12304,12290,12307,12296,12292,12294,12289,12286,12267,12289,12260,12275,12268,12266
strata cell=occ2-direct-index-resident-present-hit-replacement raw_ht_samples_ns=2914,2911,2906,2917,2904,2909,2903,2908,2904,2905,2907,2904,2909,2902,2912,2912,2905,2907,2909,2907,2911,2904,2910,2903,2905,2906,2907,2909,2900,2910,2914,2903,2912,2905,2970,2909,2906,2910,2901,2909,2904,2907,2912,2903,2910
strata cell=occ2-direct-index-resident-present-hit-wrapping-add occupancy=2 distribution=direct-index-resident operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=a436221b7910b725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12283 ht_median_ns=2906 ratio=4.2268 campaign_ratios=[4.2281,4.2207,4.2278] ratio_interval=[4.2207,4.2281] apeiron_samples_interval=[12262,14077] ht_samples_interval=[2898,2915] apeiron_ns_per_op=2.9988 ht_ns_per_op=0.7095 status=fail reason="ratio 4.2268 > 1.00"
strata cell=occ2-direct-index-resident-present-hit-wrapping-add raw_apeiron_samples_ns=12262,12301,12273,12287,12292,12291,12293,12284,12296,12283,12312,12266,12270,12268,12305,12264,12281,12274,12291,12410,14077,12351,12283,12292,12272,12278,12265,12268,12264,12268,12274,12291,12288,12295,12294,12296,12297,12286,12297,12276,12281,12267,12267,12271,12282
strata cell=occ2-direct-index-resident-present-hit-wrapping-add raw_ht_samples_ns=2909,2898,2914,2902,2911,2903,2907,2906,2905,2909,2904,2907,2905,2905,2909,2914,2902,2914,2901,2910,2903,2915,2910,2902,2910,2903,2906,2909,2904,2909,2912,2900,2910,2902,2908,2906,2904,2907,2903,2909,2904,2909,2904,2905,2907
strata cell=occ2-offset-beyond-direct-present-hit-replacement occupancy=2 distribution=offset-beyond-direct operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=361c4497f910b725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=26188 ht_median_ns=2920 ratio=8.9685 campaign_ratios=[8.9001,8.9716,9.0969] ratio_interval=[8.9001,9.0969] apeiron_samples_interval=[25817,26859] ht_samples_interval=[2906,2973] apeiron_ns_per_op=6.3936 ht_ns_per_op=0.7129 status=fail reason="ratio 8.9685 > 1.00"
strata cell=occ2-offset-beyond-direct-present-hit-replacement raw_apeiron_samples_ns=26100,26115,25987,25979,25972,26098,26106,25892,26193,25911,25920,26015,26019,25817,26272,26246,26132,26229,26101,26178,26144,26211,26131,26188,26199,26237,26314,26202,26110,26178,25919,26119,26674,26672,26390,26565,26738,26526,26465,26532,26859,26477,26563,26634,26638
strata cell=occ2-offset-beyond-direct-present-hit-replacement raw_ht_samples_ns=2928,2923,2933,2912,2925,2914,2929,2916,2918,2915,2926,2923,2920,2909,2925,2925,2906,2918,2919,2930,2917,2924,2923,2927,2920,2914,2918,2929,2918,2918,2931,2924,2926,2908,2973,2920,2920,2916,2916,2925,2926,2914,2916,2907,2919
strata cell=occ2-offset-beyond-direct-present-hit-wrapping-add occupancy=2 distribution=offset-beyond-direct operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=361c4497f910b725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=26396 ht_median_ns=2920 ratio=9.0397 campaign_ratios=[9.0175,9.2408,9.0055] ratio_interval=[9.0055,9.2408] apeiron_samples_interval=[26085,27378] ht_samples_interval=[2907,4114] apeiron_ns_per_op=6.4443 ht_ns_per_op=0.7129 status=fail reason="ratio 9.0397 > 1.00"
strata cell=occ2-offset-beyond-direct-present-hit-wrapping-add raw_apeiron_samples_ns=27074,26768,26820,26349,26085,26478,26293,26238,26396,26205,26298,26575,26422,26277,26295,26759,27011,27246,27044,27378,26736,26631,26838,26536,27116,27033,27079,26998,27159,26905,26285,26153,26205,26513,26115,26316,26331,26292,26212,26092,26269,26249,26377,26223,26340
strata cell=occ2-offset-beyond-direct-present-hit-wrapping-add raw_ht_samples_ns=2937,2914,2925,2917,2913,2909,2922,2918,2926,2907,3100,2975,2986,2939,2914,2923,2915,2972,2908,2917,2974,3005,3144,2914,4114,2935,2918,2917,2911,2973,2917,2909,2920,2948,2921,2913,2972,2917,2935,2913,2914,2928,2926,2917,2916
strata cell=occ2-seeded-random-present-hit-replacement occupancy=2 distribution=seeded-random operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=d952d2006af6d325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=25850 ht_median_ns=5596 ratio=4.6194 campaign_ratios=[4.6325,4.6210,4.5791] ratio_interval=[4.5791,4.6325] apeiron_samples_interval=[25601,26245] ht_samples_interval=[5546,8006] apeiron_ns_per_op=6.3110 ht_ns_per_op=1.3662 status=fail reason="ratio 4.6194 > 1.00"
strata cell=occ2-seeded-random-present-hit-replacement raw_apeiron_samples_ns=26080,26007,26007,26245,26096,26085,25927,26031,26050,25986,25937,26022,25929,25999,26002,25862,25821,25797,25854,25809,25822,25900,25786,25932,25881,25785,25882,25791,25909,25850,25629,25741,25691,25681,25702,25665,25619,25694,25626,25658,25661,25733,25643,25601,25620
strata cell=occ2-seeded-random-present-hit-replacement raw_ht_samples_ns=5694,5570,5954,5576,5884,5575,5899,5571,5614,5596,5846,5570,5881,5580,8006,5763,5632,5582,5576,5602,5575,5589,5604,5596,5575,5594,5703,5585,5569,5617,5872,5553,5621,5572,5876,5558,5609,5593,5606,5546,5610,5604,5919,5582,5571
strata cell=occ2-seeded-random-present-hit-wrapping-add occupancy=2 distribution=seeded-random operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=d952d2006af6d325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=25821 ht_median_ns=5577 ratio=4.6299 campaign_ratios=[4.6431,4.6048,4.6732] ratio_interval=[4.6048,4.6732] apeiron_samples_interval=[25584,26044] ht_samples_interval=[5465,7591] apeiron_ns_per_op=6.3040 ht_ns_per_op=1.3616 status=fail reason="ratio 4.6299 > 1.00"
strata cell=occ2-seeded-random-present-hit-wrapping-add raw_apeiron_samples_ns=25811,25785,25791,25890,25896,25884,25860,25805,25891,25813,25817,25891,25882,25853,25803,25660,25690,25717,25727,25788,25905,25718,25584,25764,25736,25748,25622,25735,25821,25723,25995,25976,25882,25947,25953,25957,25928,26026,25941,25920,25876,25785,26044,25908,25758
strata cell=occ2-seeded-random-present-hit-wrapping-add raw_ht_samples_ns=5557,5570,5559,5557,5605,5600,5577,5465,5580,5556,5556,5552,5583,5568,5578,5822,5546,5590,5817,5607,5617,5591,5577,5624,5527,5587,5584,5578,5573,5559,5533,5518,5546,5495,5534,5587,5581,5537,5543,5563,5604,5551,7591,5586,5630
strata cell=occ2-clustered-skewed-present-hit-replacement occupancy=2 distribution=clustered-skewed operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=c703963007559b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=26119 ht_median_ns=17669 ratio=1.4782 campaign_ratios=[1.4593,1.4972,1.4859] ratio_interval=[1.4593,1.4972] apeiron_samples_interval=[25567,28266] ht_samples_interval=[17280,20057] apeiron_ns_per_op=6.3767 ht_ns_per_op=4.3137 status=fail reason="ratio 1.4782 > 1.00"
strata cell=occ2-clustered-skewed-present-hit-replacement raw_apeiron_samples_ns=27473,26202,26392,26367,26280,26184,28266,25621,25567,25778,25639,25761,25588,25804,25697,25883,28161,26392,26557,27197,26707,26704,26451,26463,26659,26557,26960,27720,26397,26470,25927,25874,26080,26131,25997,26041,26119,25810,25847,25937,25944,25965,26108,25852,26034
strata cell=occ2-clustered-skewed-present-hit-replacement raw_ht_samples_ns=17705,17487,17893,17511,17645,17941,20057,17525,18071,17582,17943,17662,17682,17707,17683,17557,17926,17982,18423,17889,17685,18562,17738,19639,17575,17470,17899,17613,17669,17543,17920,18001,17453,17593,17460,17525,17280,17474,17362,18852,17296,17555,17410,17453,17981
strata cell=occ2-clustered-skewed-present-hit-wrapping-add occupancy=2 distribution=clustered-skewed operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=c703963007559b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=26142 ht_median_ns=17568 ratio=1.4880 campaign_ratios=[1.4784,1.5153,1.4693] ratio_interval=[1.4693,1.5153] apeiron_samples_interval=[25456,28574] ht_samples_interval=[17189,19887] apeiron_ns_per_op=6.3823 ht_ns_per_op=4.2891 status=fail reason="ratio 1.4880 > 1.00"
strata cell=occ2-clustered-skewed-present-hit-wrapping-add raw_apeiron_samples_ns=26265,25789,25745,26183,26050,26083,26142,26101,26153,25992,25971,26138,26055,26285,26402,26563,26539,26293,26859,26709,26713,26620,26646,26718,26760,28574,26369,26215,26242,26081,26246,26258,26220,26090,25872,25942,25707,25456,25584,25939,25627,25532,25596,25650,25691
strata cell=occ2-clustered-skewed-present-hit-wrapping-add raw_ht_samples_ns=17804,17560,19887,17813,17655,17665,17661,17676,17742,17534,17655,17623,17554,17531,17505,18199,17520,17449,17872,17567,17477,17746,17521,17796,17414,19027,17566,19498,17924,17568,18129,17431,17470,18500,17538,18571,19573,17351,17602,17189,17522,17496,17205,17408,17492
strata cell=occ4-direct-index-resident-present-hit-replacement occupancy=4 distribution=direct-index-resident operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=05920a92a50ba725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12283 ht_median_ns=2908 ratio=4.2239 campaign_ratios=[4.2275,4.2246,4.2224] ratio_interval=[4.2224,4.2275] apeiron_samples_interval=[12261,13758] ht_samples_interval=[2897,2980] apeiron_ns_per_op=2.9988 ht_ns_per_op=0.7100 status=fail reason="ratio 4.2239 > 1.00"
strata cell=occ4-direct-index-resident-present-hit-replacement raw_apeiron_samples_ns=13758,12277,12262,12300,12300,12285,12266,12296,12314,12275,12267,12300,12295,12263,12273,12266,12284,12298,12273,12268,12281,12296,12279,12274,12295,12297,12270,12261,12292,12296,12284,12291,12277,12266,12283,12291,12275,12286,12282,12290,12333,12270,12283,12291,12278
strata cell=occ4-direct-index-resident-present-hit-replacement raw_ht_samples_ns=2915,2918,2910,2906,2903,2905,2935,2901,2901,2903,2910,2902,2904,2907,2911,2906,2897,2908,2911,2907,2899,2908,2911,2906,2909,2905,2909,2910,2903,2905,2910,2909,2902,2900,2909,2910,2902,2899,2911,2980,2928,2920,2909,2908,2903
strata cell=occ4-direct-index-resident-present-hit-wrapping-add occupancy=4 distribution=direct-index-resident operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=05920a92a50ba725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12325 ht_median_ns=2912 ratio=4.2325 campaign_ratios=[4.2292,4.2339,4.2357] ratio_interval=[4.2292,4.2357] apeiron_samples_interval=[12297,14302] ht_samples_interval=[2903,3023] apeiron_ns_per_op=3.0090 ht_ns_per_op=0.7109 status=fail reason="ratio 4.2325 > 1.00"
strata cell=occ4-direct-index-resident-present-hit-wrapping-add raw_apeiron_samples_ns=12321,12332,12316,12313,12324,12331,12319,12309,12329,12326,12344,12309,12329,12325,12307,12333,12302,12315,12333,12325,12302,12383,12354,12316,12309,12332,12333,12314,12304,12326,12306,12330,12334,12297,12318,12332,12332,12303,12305,12332,12334,12303,12307,12336,14302
strata cell=occ4-direct-index-resident-present-hit-wrapping-add raw_ht_samples_ns=2919,2948,2911,2912,2918,2914,2908,2905,2914,2914,2912,2903,2916,2914,2912,2910,2908,2919,2913,2907,2911,2920,2915,2910,2909,2918,2916,2909,2907,2919,2919,2908,2908,2912,2917,3023,2908,2911,2918,2907,2909,2915,2917,2907,2911
strata cell=occ4-offset-beyond-direct-present-hit-replacement occupancy=4 distribution=offset-beyond-direct operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=ae36ace7470ba725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=24712 ht_median_ns=3056 ratio=8.0864 campaign_ratios=[8.0613,8.1001,8.0792] ratio_interval=[8.0613,8.1001] apeiron_samples_interval=[24624,26343] ht_samples_interval=[3036,3172] apeiron_ns_per_op=6.0332 ht_ns_per_op=0.7461 status=fail reason="ratio 8.0864 > 1.00"
strata cell=occ4-offset-beyond-direct-present-hit-replacement raw_apeiron_samples_ns=24704,24734,24723,24712,24714,24716,24680,24657,24691,24715,25571,25044,24993,25026,25106,24800,24754,24818,24741,24768,24800,24786,26343,24714,24682,24711,24760,24683,24707,24683,24698,24682,24707,24711,24707,24733,24716,24673,24624,24664,24643,24655,24702,24659,24626
strata cell=occ4-offset-beyond-direct-present-hit-replacement raw_ht_samples_ns=3072,3047,3066,3059,3097,3075,3092,3045,3067,3144,3075,3045,3061,3038,3049,3071,3037,3050,3040,3060,3042,3063,3039,3172,3056,3097,3045,3056,3036,3087,3064,3039,3056,3063,3072,3055,3053,3043,3062,3053,3065,3054,3051,3048,3081
strata cell=occ4-offset-beyond-direct-present-hit-wrapping-add occupancy=4 distribution=offset-beyond-direct operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=ae36ace7470ba725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=24811 ht_median_ns=3050 ratio=8.1348 campaign_ratios=[8.1379,8.0886,8.1761] ratio_interval=[8.0886,8.1761] apeiron_samples_interval=[24283,25077] ht_samples_interval=[3031,3099] apeiron_ns_per_op=6.0574 ht_ns_per_op=0.7446 status=fail reason="ratio 8.1348 > 1.00"
strata cell=occ4-offset-beyond-direct-present-hit-wrapping-add raw_apeiron_samples_ns=24624,24758,24662,24643,24837,24827,24811,25017,25077,24963,25005,24929,24946,24951,24845,24624,24637,24671,24729,24687,24712,24626,24572,24614,24646,24634,24662,24678,24603,24890,24942,24975,24936,24966,24947,24937,24851,24973,24876,24960,24941,24745,24738,24352,24283
strata cell=occ4-offset-beyond-direct-present-hit-wrapping-add raw_ht_samples_ns=3059,3050,3057,3035,3053,3041,3064,3042,3049,3031,3054,3042,3072,3094,3060,3064,3045,3056,3039,3047,3037,3064,3044,3062,3037,3047,3039,3058,3051,3099,3054,3037,3064,3036,3050,3045,3043,3042,3058,3046,3056,3053,3054,3039,3056
strata cell=occ4-seeded-random-present-hit-replacement occupancy=4 distribution=seeded-random operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=5647b1a930287725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=25942 ht_median_ns=5986 ratio=4.3338 campaign_ratios=[4.1763,4.3308,4.3925] ratio_interval=[4.1763,4.3925] apeiron_samples_interval=[25552,28176] ht_samples_interval=[5587,6748] apeiron_ns_per_op=6.3335 ht_ns_per_op=1.4614 status=fail reason="ratio 4.3338 > 1.00"
strata cell=occ4-seeded-random-present-hit-replacement raw_apeiron_samples_ns=26911,26871,26402,26319,26062,26500,26262,26464,25780,26156,25881,26026,25950,25942,25966,28176,26061,25750,25736,25552,25829,25610,25861,25716,25644,25846,25777,25644,27613,27067,25866,26046,25981,25936,26003,25980,25777,25901,25832,27375,25795,25683,25694,25985,25907
strata cell=occ4-seeded-random-present-hit-replacement raw_ht_samples_ns=6748,6463,6463,6263,5989,5743,5861,6660,6377,6146,6212,5955,6333,6176,6322,6739,6002,6085,5879,6046,5952,5975,5892,6004,5929,5951,5894,5934,5887,5993,5962,6085,5856,5898,5904,5720,5722,5741,5685,5587,5986,5686,6103,6051,6171
strata cell=occ4-seeded-random-present-hit-wrapping-add occupancy=4 distribution=seeded-random operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=5647b1a930287725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=26012 ht_median_ns=5910 ratio=4.4014 campaign_ratios=[4.3720,4.3740,4.4747] ratio_interval=[4.3720,4.4747] apeiron_samples_interval=[25689,27468] ht_samples_interval=[5588,7592] apeiron_ns_per_op=6.3506 ht_ns_per_op=1.4429 status=fail reason="ratio 4.4014 > 1.00"
strata cell=occ4-seeded-random-present-hit-wrapping-add raw_apeiron_samples_ns=26126,26089,25909,26101,25995,26559,26184,26454,26145,26382,27468,26482,26221,26181,26449,26199,26134,25978,26103,26012,25980,25921,26066,25861,26164,25893,26250,25943,26147,25893,26197,25905,25839,25940,25869,26011,25963,26087,25959,25886,25967,25972,25689,25914,25785
strata cell=occ4-seeded-random-present-hit-wrapping-add raw_ht_samples_ns=5989,6051,6269,7592,5945,5945,5873,5739,5846,5952,6149,5635,6496,6040,6003,6086,5843,5847,5917,5913,5885,6078,6134,5972,5947,5994,5910,5972,5789,6002,5891,5797,5765,5746,5806,5588,5797,5826,5852,5734,5852,5607,5911,5773,5797
strata cell=occ4-clustered-skewed-present-hit-replacement occupancy=4 distribution=clustered-skewed operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=89d5cc5ff13ced25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=29789 ht_median_ns=22358 ratio=1.3324 campaign_ratios=[1.3744,1.3285,1.3055] ratio_interval=[1.3055,1.3744] apeiron_samples_interval=[28517,36824] ht_samples_interval=[21628,25001] apeiron_ns_per_op=7.2727 ht_ns_per_op=5.4585 status=fail reason="ratio 1.3324 > 1.00"
strata cell=occ4-clustered-skewed-present-hit-replacement raw_apeiron_samples_ns=31987,31170,30606,30090,29914,29754,29453,36824,30859,31130,30709,30351,30032,30083,31438,29623,29928,29765,29408,29311,29897,29789,35281,31248,30824,30101,30126,29260,29483,29276,29670,29802,29057,29761,29302,29490,34387,29770,29163,29189,28700,28581,28517,28961,28776
strata cell=occ4-clustered-skewed-present-hit-replacement raw_ht_samples_ns=24885,22104,24678,22268,22803,22150,22385,21628,23180,21894,22709,22236,22192,22040,22502,22228,22100,22584,22049,25001,21681,22451,23374,23190,21794,22348,23533,22423,21849,22428,24520,21958,22305,21633,22358,22015,24496,21789,22729,21658,23617,21675,22742,22520,22731
strata cell=occ4-clustered-skewed-present-hit-wrapping-add occupancy=4 distribution=clustered-skewed operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=89d5cc5ff13ced25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=29402 ht_median_ns=22400 ratio=1.3126 campaign_ratios=[1.3358,1.2981,1.3076] ratio_interval=[1.2981,1.3358] apeiron_samples_interval=[27850,51806] ht_samples_interval=[21635,24857] apeiron_ns_per_op=7.1782 ht_ns_per_op=5.4688 status=fail reason="ratio 1.3126 > 1.00"
strata cell=occ4-clustered-skewed-present-hit-wrapping-add raw_apeiron_samples_ns=28143,27850,28225,28057,28306,36805,30531,29908,30192,30226,29938,30321,29859,29662,30883,29486,29056,29090,29132,30703,28854,29150,29205,33532,29811,29664,29261,29402,28628,29029,29184,28843,51806,34484,29697,29400,29604,29064,29229,28908,29297,29367,33771,31466,30602
strata cell=occ4-clustered-skewed-present-hit-wrapping-add raw_ht_samples_ns=22031,21755,22289,21730,22516,24857,23705,22390,22834,22044,22484,22400,22722,21950,22295,22076,21805,22683,21708,22499,22372,22871,22012,24609,22345,23251,22611,24606,22741,22252,21635,21650,22289,23481,22527,21778,23428,21819,22620,21808,24017,21806,24632,22484,22825
strata cell=occ8-direct-index-resident-present-hit-replacement occupancy=8 distribution=direct-index-resident operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=cdc34167bb14db25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12290 ht_median_ns=2909 ratio=4.2248 campaign_ratios=[4.2248,4.2256,4.2223] ratio_interval=[4.2223,4.2256] apeiron_samples_interval=[12260,13835] ht_samples_interval=[2899,2927] apeiron_ns_per_op=3.0005 ht_ns_per_op=0.7102 status=fail reason="ratio 4.2248 > 1.00"
strata cell=occ8-direct-index-resident-present-hit-replacement raw_apeiron_samples_ns=13835,12288,12290,12292,12282,12268,12331,12289,12311,12307,12277,12359,12463,12279,12285,12277,12300,12288,12338,12301,12291,12271,12312,12263,12263,12292,12272,12279,12297,12262,12260,12275,12297,12383,12337,12301,12332,12264,12291,12272,12281,12294,12263,12264,12297
strata cell=occ8-direct-index-resident-present-hit-replacement raw_ht_samples_ns=2920,2908,2912,2916,2915,2910,2906,2901,2911,2907,2924,2909,2906,2899,2909,2907,2908,2927,2909,2903,2901,2910,2909,2915,2910,2907,2901,2907,2906,2918,2917,2915,2910,2908,2908,2912,2917,2912,2911,2903,2908,2909,2915,2913,2906
strata cell=occ8-direct-index-resident-present-hit-wrapping-add occupancy=8 distribution=direct-index-resident operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=cdc34167bb14db25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12329 ht_median_ns=2914 ratio=4.2310 campaign_ratios=[4.2299,4.2252,4.2345] ratio_interval=[4.2252,4.2345] apeiron_samples_interval=[12302,13842] ht_samples_interval=[2906,2920] apeiron_ns_per_op=3.0100 ht_ns_per_op=0.7114 status=fail reason="ratio 4.2310 > 1.00"
strata cell=occ8-direct-index-resident-present-hit-wrapping-add raw_apeiron_samples_ns=12317,12331,12326,12302,12382,12342,12310,12315,12336,12347,12313,12336,12308,12305,12337,12336,12345,12308,12316,12336,12315,12325,12371,12310,12311,12344,12338,12309,12325,12329,12348,12343,12321,12342,12334,12304,12338,13842,12305,12320,12362,12308,12335,12363,12305
strata cell=occ8-direct-index-resident-present-hit-wrapping-add raw_ht_samples_ns=2915,2916,2918,2911,2915,2907,2920,2914,2912,2907,2913,2914,2920,2914,2912,2909,2907,2919,2917,2913,2909,2919,2918,2920,2915,2914,2908,2917,2918,2918,2912,2906,2912,2913,2919,2913,2911,2910,2919,2918,2915,2913,2910,2914,2918
strata cell=occ8-offset-beyond-direct-present-hit-replacement occupancy=8 distribution=offset-beyond-direct operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=91027062e914db25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=25078 ht_median_ns=3086 ratio=8.1264 campaign_ratios=[8.1491,8.0376,8.1250] ratio_interval=[8.0376,8.1491] apeiron_samples_interval=[24743,26689] ht_samples_interval=[3056,3238] apeiron_ns_per_op=6.1226 ht_ns_per_op=0.7534 status=fail reason="ratio 8.1264 > 1.00"
strata cell=occ8-offset-beyond-direct-present-hit-replacement raw_apeiron_samples_ns=25038,24949,25098,25140,25121,26515,25258,25168,26689,25173,25218,25169,25036,24977,24981,24804,24800,24777,24812,24785,24886,24873,24804,24799,24835,24826,24842,25143,24791,24743,25082,25104,25080,25039,25070,25073,25081,25081,25179,25162,25183,25155,25194,25095,25078
strata cell=occ8-offset-beyond-direct-present-hit-replacement raw_ht_samples_ns=3072,3149,3111,3238,3085,3232,3147,3067,3078,3056,3080,3061,3189,3165,3069,3086,3092,3087,3094,3082,3186,3081,3072,3081,3162,3076,3155,3154,3077,3071,3096,3087,3093,3060,3083,3057,3092,3078,3097,3090,3091,3066,3102,3066,3074
strata cell=occ8-offset-beyond-direct-present-hit-wrapping-add occupancy=8 distribution=offset-beyond-direct operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=91027062e914db25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=24889 ht_median_ns=3081 ratio=8.0782 campaign_ratios=[8.0747,8.1108,8.0427] ratio_interval=[8.0427,8.1108] apeiron_samples_interval=[24749,25185] ht_samples_interval=[3052,3194] apeiron_ns_per_op=6.0764 ht_ns_per_op=0.7522 status=fail reason="ratio 8.0782 > 1.00"
strata cell=occ8-offset-beyond-direct-present-hit-wrapping-add raw_apeiron_samples_ns=24860,24809,24752,24833,24768,24846,24807,24782,24749,25121,25185,25180,25063,25059,25049,25030,25036,25051,25053,25006,25013,25028,25031,25062,25018,25110,25063,24858,24828,24800,24829,24790,24845,24852,24752,24843,24861,24765,24782,24899,24924,24929,24857,24889,24966
strata cell=occ8-offset-beyond-direct-present-hit-wrapping-add raw_ht_samples_ns=3118,3073,3171,3067,3079,3052,3086,3060,3078,3073,3077,3069,3096,3072,3088,3100,3173,3110,3073,3086,3071,3086,3072,3080,3066,3081,3059,3194,3094,3120,3098,3098,3090,3115,3134,3069,3081,3067,3091,3069,3071,3056,3175,3143,3086
strata cell=occ8-seeded-random-present-hit-replacement occupancy=8 distribution=seeded-random operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=816aec4d94518725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=30629 ht_median_ns=6627 ratio=4.6219 campaign_ratios=[4.5841,4.7284,4.6824] ratio_interval=[4.5841,4.7284] apeiron_samples_interval=[29637,35372] ht_samples_interval=[5954,7938] apeiron_ns_per_op=7.4778 ht_ns_per_op=1.6179 status=fail reason="ratio 4.6219 > 1.00"
strata cell=occ8-seeded-random-present-hit-replacement raw_apeiron_samples_ns=32286,31281,30072,32134,29995,30391,30477,30629,30220,30379,30577,29873,29637,29782,29878,30359,33770,31303,30685,30498,30688,30440,30490,30206,30572,29755,30282,30072,34234,34938,34178,31344,30361,35372,32236,31780,30824,31010,30693,34492,32189,31737,30609,31205,30872
strata cell=occ8-seeded-random-present-hit-replacement raw_ht_samples_ns=7542,7011,6749,6136,6406,6202,6627,6256,6325,6650,7403,6854,6506,6029,6699,6713,7305,6735,6152,6450,6298,6545,6161,6596,6388,6160,5954,6223,7107,7938,7476,6862,6734,7422,6947,6649,6890,6761,6694,6442,6228,6109,6324,6575,6675
strata cell=occ8-seeded-random-present-hit-wrapping-add occupancy=8 distribution=seeded-random operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=816aec4d94518725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=30781 ht_median_ns=6806 ratio=4.5226 campaign_ratios=[4.6153,4.5065,4.5259] ratio_interval=[4.5065,4.6153] apeiron_samples_interval=[29840,36211] ht_samples_interval=[6272,8792] apeiron_ns_per_op=7.5149 ht_ns_per_op=1.6616 status=fail reason="ratio 4.5226 > 1.00"
strata cell=occ8-seeded-random-present-hit-wrapping-add raw_apeiron_samples_ns=33403,31784,30278,30504,29840,33716,30992,30789,29959,30338,29990,36211,34343,32006,31035,32794,31044,30156,30724,30282,30783,30519,30687,30236,30649,33043,31151,30303,30471,29993,30781,36054,31139,30555,30194,33976,31373,30980,30314,31363,32756,31243,30420,30437,30015
strata cell=occ8-seeded-random-present-hit-wrapping-add raw_ht_samples_ns=7494,6715,6806,6495,8792,7125,7282,7032,6503,6481,7021,6635,6669,6497,6546,7829,6947,7049,6460,6462,6272,6583,6757,6982,6995,7704,6801,6639,6593,6872,6589,6872,6505,6522,6664,7437,7001,6486,6966,6553,7191,6846,6831,6845,6887
strata cell=occ8-clustered-skewed-present-hit-replacement occupancy=8 distribution=clustered-skewed operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=77910d7a944e0925 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=51150 ht_median_ns=28629 ratio=1.7866 campaign_ratios=[1.8305,1.7917,1.7828] ratio_interval=[1.7828,1.8305] apeiron_samples_interval=[46933,62133] ht_samples_interval=[27434,30989] apeiron_ns_per_op=12.4878 ht_ns_per_op=6.9895 status=fail reason="ratio 1.7866 > 1.00"
strata cell=occ8-clustered-skewed-present-hit-replacement raw_apeiron_samples_ns=54315,48391,46933,46979,47063,56240,49667,48501,59821,52656,50907,60594,54142,62133,54340,58033,51351,49099,48806,49138,48466,48343,60479,53552,51401,60054,55973,53337,51857,51150,58590,51118,50655,48275,47903,52144,49526,49033,61243,55561,51252,50327,50518,50251,49749
strata cell=occ8-clustered-skewed-present-hit-replacement raw_ht_samples_ns=30203,30757,28766,27860,28459,30989,28634,27978,30246,29421,29341,28230,29107,27434,28220,30430,29046,28762,28022,28689,28265,29068,27770,28629,27874,30318,29061,28537,27895,29425,30266,27745,28184,27734,27852,28492,28118,27906,30488,28887,28876,27876,28540,28337,28773
strata cell=occ8-clustered-skewed-present-hit-wrapping-add occupancy=8 distribution=clustered-skewed operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=77910d7a944e0925 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=53530 ht_median_ns=28802 ratio=1.8586 campaign_ratios=[1.8637,1.8768,1.8463] ratio_interval=[1.8463,1.8768] apeiron_samples_interval=[49191,64206] ht_samples_interval=[27263,31250] apeiron_ns_per_op=13.0688 ht_ns_per_op=7.0317 status=fail reason="ratio 1.8586 > 1.00"
strata cell=occ8-clustered-skewed-present-hit-wrapping-add raw_apeiron_samples_ns=49191,49503,60217,55290,51575,50415,60105,54339,52719,52117,61403,54292,60151,53746,51127,57275,50715,60666,54425,52254,52099,51381,59955,53399,51681,61523,54056,51683,64206,55367,59003,50491,49897,49371,61258,53530,51988,60459,55736,52968,51393,52046,62840,53792,51921
strata cell=occ8-clustered-skewed-present-hit-wrapping-add raw_ht_samples_ns=29277,27679,30698,29176,28717,28181,30737,28635,28839,28646,30909,30002,30143,28665,28485,30128,28802,30368,29215,28774,28582,28945,27737,28642,28098,30561,29391,29560,28095,27877,30452,28672,28570,28224,30334,28923,29122,27780,28170,28098,28688,27263,31250,29098,29060
strata cell=occ16-direct-index-resident-present-hit-replacement occupancy=16 distribution=direct-index-resident operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=80eb180389314b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12285 ht_median_ns=2919 ratio=4.2086 campaign_ratios=[4.2164,4.2093,4.2003] ratio_interval=[4.2003,4.2164] apeiron_samples_interval=[12258,13225] ht_samples_interval=[2910,3284] apeiron_ns_per_op=2.9993 ht_ns_per_op=0.7126 status=fail reason="ratio 4.2086 > 1.00"
strata cell=occ16-direct-index-resident-present-hit-replacement raw_apeiron_samples_ns=12284,12294,12426,13225,12297,12421,12297,12294,12325,12288,12294,12282,12295,12285,12296,12336,12274,12287,12293,12302,12267,12289,12265,12288,12273,12307,12260,12297,12265,12279,12274,12262,12276,12258,12279,12296,12268,12290,12262,12261,12278,12266,12265,12272,12269
strata cell=occ16-direct-index-resident-present-hit-replacement raw_ht_samples_ns=2914,2912,3016,3284,2916,2923,2916,2917,2915,2918,2917,2915,2912,2915,2919,2912,2923,2923,2919,2918,2921,2919,2921,2918,2919,2925,2919,2926,2922,2919,2921,2918,2921,2927,2920,2921,2926,2930,2927,2910,2920,2913,2921,2923,2922
strata cell=occ16-direct-index-resident-present-hit-wrapping-add occupancy=16 distribution=direct-index-resident operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=80eb180389314b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=12271 ht_median_ns=2921 ratio=4.2010 campaign_ratios=[4.2063,4.1931,4.1974] ratio_interval=[4.1931,4.2063] apeiron_samples_interval=[12258,12308] ht_samples_interval=[2904,2959] apeiron_ns_per_op=2.9958 ht_ns_per_op=0.7131 status=fail reason="ratio 4.2010 > 1.00"
strata cell=occ16-direct-index-resident-present-hit-wrapping-add raw_apeiron_samples_ns=12272,12291,12272,12292,12271,12291,12274,12301,12265,12288,12267,12288,12268,12282,12263,12308,12286,12301,12279,12258,12272,12258,12271,12264,12268,12265,12275,12269,12266,12261,12266,12270,12264,12260,12294,12280,12261,12272,12308,12269,12268,12265,12277,12264,12273
strata cell=occ16-direct-index-resident-present-hit-wrapping-add raw_ht_samples_ns=2917,2918,2919,2923,2921,2904,2919,2915,2916,2906,2919,2906,2928,2916,2930,2933,2912,2926,2919,2926,2917,2933,2913,2932,2923,2927,2959,2929,2912,2930,2933,2923,2934,2914,2937,2915,2930,2918,2921,2921,2924,2915,2926,2917,2931
strata cell=occ16-offset-beyond-direct-present-hit-replacement occupancy=16 distribution=offset-beyond-direct operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=baeed7906b314b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=26471 ht_median_ns=3238 ratio=8.1751 campaign_ratios=[8.1433,8.1683,8.2098] ratio_interval=[8.1433,8.2098] apeiron_samples_interval=[26311,27663] ht_samples_interval=[3204,3362] apeiron_ns_per_op=6.4626 ht_ns_per_op=0.7905 status=fail reason="ratio 8.1751 > 1.00"
strata cell=occ16-offset-beyond-direct-present-hit-replacement raw_apeiron_samples_ns=26371,26311,26344,26497,26530,26472,26400,26400,26682,26336,26350,26376,26355,26335,27640,26449,26428,26432,27663,26463,27522,27548,27497,27477,27562,26356,26394,26418,26399,26400,26629,26603,26525,26543,26543,26686,26662,26345,26404,26334,26471,26681,26583,26567,26623
strata cell=occ16-offset-beyond-direct-present-hit-replacement raw_ht_samples_ns=3345,3219,3333,3250,3237,3218,3248,3226,3263,3239,3234,3209,3243,3217,3344,3238,3217,3318,3238,3362,3218,3245,3230,3240,3224,3257,3210,3240,3204,3238,3245,3206,3229,3211,3236,3315,3226,3217,3320,3224,3243,3318,3213,3238,3245
strata cell=occ16-offset-beyond-direct-present-hit-wrapping-add occupancy=16 distribution=offset-beyond-direct operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=baeed7906b314b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=26651 ht_median_ns=3244 ratio=8.2155 campaign_ratios=[8.1742,8.2394,8.1643] ratio_interval=[8.1643,8.2394] apeiron_samples_interval=[26247,27923] ht_samples_interval=[3201,4347] apeiron_ns_per_op=6.5066 ht_ns_per_op=0.7920 status=fail reason="ratio 8.2155 > 1.00"
strata cell=occ16-offset-beyond-direct-present-hit-wrapping-add raw_apeiron_samples_ns=26727,26705,26651,26683,26460,26319,26307,26335,26300,26361,27923,27390,26247,26591,26407,27095,26998,27036,27345,27248,27058,27042,27017,27069,26987,26311,26413,26714,26404,26471,26278,26762,26781,27169,27476,26773,26797,26608,26393,26363,26395,26421,26390,26480,26485
strata cell=occ16-offset-beyond-direct-present-hit-wrapping-add raw_ht_samples_ns=3317,3230,3238,3208,3261,3311,3237,3214,3236,3225,3313,3482,3393,3209,3229,3241,3212,3533,3369,3307,3229,3308,3283,3279,3232,3243,3225,3378,3285,3254,3225,3391,3283,3297,4347,3305,3231,3251,3231,3201,3227,3239,3218,3309,3244
strata cell=occ16-seeded-random-present-hit-replacement occupancy=16 distribution=seeded-random operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=941af7cf934ad325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=43431 ht_median_ns=6340 ratio=6.8503 campaign_ratios=[6.9784,6.6139,6.7659] ratio_interval=[6.6139,6.9784] apeiron_samples_interval=[39917,47755] ht_samples_interval=[5887,8617] apeiron_ns_per_op=10.6033 ht_ns_per_op=1.5479 status=fail reason="ratio 6.8503 > 1.00"
strata cell=occ16-seeded-random-present-hit-replacement raw_apeiron_samples_ns=46584,43343,47002,43319,44126,47338,45259,43758,42321,47755,44243,46519,43899,41791,44870,46377,43158,43997,47692,44099,43750,42707,41450,41853,41610,40730,39917,41562,41012,40794,46838,43794,41744,42786,41653,43612,43383,40950,45813,43431,41411,44293,41216,41694,45050
strata cell=occ16-seeded-random-present-hit-replacement raw_ht_samples_ns=6615,6313,6855,6346,6307,6016,6536,6382,6340,6329,6055,6677,6206,6005,6522,6542,6328,6456,6022,5887,5997,8617,6312,6914,6323,6337,6312,6480,6161,6475,6635,6286,6687,6471,6352,6252,6096,5944,6645,6412,6724,6566,6150,6132,6576
strata cell=occ16-seeded-random-present-hit-wrapping-add occupancy=16 distribution=seeded-random operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=941af7cf934ad325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=43553 ht_median_ns=6283 ratio=6.9319 campaign_ratios=[6.9350,6.9347,7.2755] ratio_interval=[6.9347,7.2755] apeiron_samples_interval=[41617,47642] ht_samples_interval=[5956,8113] apeiron_ns_per_op=10.6331 ht_ns_per_op=1.5339 status=fail reason="ratio 6.9319 > 1.00"
strata cell=occ16-seeded-random-present-hit-wrapping-add raw_apeiron_samples_ns=45442,42385,45755,42998,43553,41974,41684,45809,43614,45058,43262,46933,44310,41808,44152,46648,43085,41842,41905,47642,43800,43146,42883,46448,45527,42515,41648,41768,41617,44691,46232,43502,42900,45994,43227,45835,42653,41888,41774,46227,45309,46595,45021,45764,42458
strata cell=occ16-seeded-random-present-hit-wrapping-add raw_ht_samples_ns=6569,6289,6550,6234,6184,6075,6039,6283,6362,6143,6464,6305,7776,6054,6579,8113,6502,6160,6150,6619,6356,6180,6213,6533,6528,6090,5980,6108,6069,6535,6719,6374,6313,6315,6088,6179,5961,5956,6428,6188,6397,6104,6181,6358,6121
strata cell=occ16-clustered-skewed-present-hit-replacement occupancy=16 distribution=clustered-skewed operation=present-hit merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=511cff563a411b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=71277 ht_median_ns=19754 ratio=3.6082 campaign_ratios=[3.6450,3.6039,3.6039] ratio_interval=[3.6039,3.6450] apeiron_samples_interval=[69942,75000] ht_samples_interval=[19596,22516] apeiron_ns_per_op=17.4016 ht_ns_per_op=4.8228 status=fail reason="ratio 3.6082 > 1.00"
strata cell=occ16-clustered-skewed-present-hit-replacement raw_apeiron_samples_ns=72000,72858,72124,71156,72046,70454,69942,74731,71082,70370,73670,74784,71701,70525,74722,71205,70038,70554,72758,72191,72101,71705,70552,72523,73031,70603,71161,71197,71058,73876,72020,70216,73278,70206,71277,70863,72067,71192,71081,73474,70596,70230,71340,70249,75000
strata cell=occ16-clustered-skewed-present-hit-replacement raw_ht_samples_ns=19702,19692,19668,19673,19927,19821,19785,20322,20084,19714,19657,19753,19715,19871,19845,19754,19856,19677,19714,22516,20027,19734,19784,19724,19752,19758,20600,19720,19795,19921,19859,19831,19732,19754,19760,19717,19732,20074,21149,19724,19596,19681,20025,19733,19773
strata cell=occ16-clustered-skewed-present-hit-wrapping-add occupancy=16 distribution=clustered-skewed operation=present-hit merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=511cff563a411b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=71500 ht_median_ns=19816 ratio=3.6082 campaign_ratios=[3.6272,3.5885,3.6059] ratio_interval=[3.5885,3.6272] apeiron_samples_interval=[68700,74892] ht_samples_interval=[19710,20370] apeiron_ns_per_op=17.4561 ht_ns_per_op=4.8379 status=fail reason="ratio 3.6082 > 1.00"
strata cell=occ16-clustered-skewed-present-hit-wrapping-add raw_apeiron_samples_ns=71739,71627,72475,71025,74858,71882,72097,71876,70307,71469,72838,73490,70565,71500,70246,72158,71663,71179,70908,74892,72248,71510,71182,72598,68700,71749,70863,70945,69872,69971,72320,71986,70401,71455,71676,71495,71899,70505,71982,70822,71228,70656,69491,72259,71365
strata cell=occ16-clustered-skewed-present-hit-wrapping-add raw_ht_samples_ns=19778,19823,19722,20254,19869,19770,19737,19767,19710,19763,19860,19883,20153,19752,19860,19745,19762,19902,19836,19774,19951,19857,19866,19790,19789,19867,19982,19845,19758,19756,19790,19839,19796,20370,20232,19718,19897,19842,19814,19806,19992,19904,19726,19763,19816
strata cell=occ1-direct-index-resident-absent-miss-replacement occupancy=1 distribution=direct-index-resident operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=701ea041d2d7f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10850 ht_median_ns=2183 ratio=4.9702 campaign_ratios=[4.9743,4.9734,4.9698] ratio_interval=[4.9698,4.9743] apeiron_samples_interval=[10820,13730] ht_samples_interval=[2177,4381] apeiron_ns_per_op=2.6489 ht_ns_per_op=0.5330 status=fail reason="ratio 4.9702 > 1.00"
strata cell=occ1-direct-index-resident-absent-miss-replacement raw_apeiron_samples_ns=10904,10911,10847,10874,10867,10831,10825,10837,10878,10854,10865,10860,10846,10838,10826,10850,10853,10863,13730,10856,10843,10830,10827,10842,10843,10864,10861,10855,10852,10831,10841,10842,10867,10901,10852,10873,10831,10820,10877,10833,10849,10851,10849,10846,10826
strata cell=occ1-direct-index-resident-absent-miss-replacement raw_ht_samples_ns=2191,2184,2185,2179,4381,2179,2182,2181,2182,2185,2181,2179,2185,2187,2182,2178,2183,2183,2177,2184,2177,2183,2180,2181,2184,2181,2182,2187,2178,2185,2178,2190,2183,2183,2188,2177,2182,2185,2180,2187,2188,2181,2187,2180,2183
strata cell=occ1-direct-index-resident-absent-miss-wrapping-add occupancy=1 distribution=direct-index-resident operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=701ea041d2d7f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10842 ht_median_ns=2178 ratio=4.9780 campaign_ratios=[4.9812,4.9752,4.9711] ratio_interval=[4.9711,4.9812] apeiron_samples_interval=[10817,12869] ht_samples_interval=[2173,2218] apeiron_ns_per_op=2.6470 ht_ns_per_op=0.5317 status=fail reason="ratio 4.9780 > 1.00"
strata cell=occ1-direct-index-resident-absent-miss-wrapping-add raw_apeiron_samples_ns=10824,12032,10827,10829,10851,10854,10846,10844,10828,10821,10826,10832,10854,10852,10849,10823,10819,10834,10836,10856,10853,12869,10842,10822,10822,10824,10831,10853,10854,10847,10817,10819,10836,10842,10850,10855,10836,10832,11010,12026,10826,10832,10848,10852,10848
strata cell=occ1-direct-index-resident-absent-miss-wrapping-add raw_ht_samples_ns=2179,2177,2176,2180,2175,2181,2180,2176,2177,2176,2174,2183,2173,2180,2180,2179,2178,2173,2184,2176,2178,2191,2177,2181,2178,2176,2183,2177,2179,2180,2178,2177,2178,2183,2185,2178,2183,2181,2216,2218,2173,2182,2177,2177,2181
strata cell=occ1-offset-beyond-direct-absent-miss-replacement occupancy=1 distribution=offset-beyond-direct operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=7c0c39940cd7f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=46894 ht_median_ns=9215 ratio=5.0889 campaign_ratios=[4.7410,5.0974,5.2916] ratio_interval=[4.7410,5.2916] apeiron_samples_interval=[46452,49187] ht_samples_interval=[8716,11821] apeiron_ns_per_op=11.4487 ht_ns_per_op=2.2498 status=fail reason="ratio 5.0889 > 1.00"
strata cell=occ1-offset-beyond-direct-absent-miss-replacement raw_apeiron_samples_ns=46637,47034,46979,46936,46744,46676,46922,46843,46466,46752,46554,46932,47658,49187,47203,46730,46978,47249,47069,46772,46912,46722,46689,46623,48688,47172,47099,46799,46456,46718,46452,47013,47052,46860,46698,46996,48893,46894,46957,46758,46773,46763,46760,47945,47071
strata cell=occ1-offset-beyond-direct-absent-miss-replacement raw_ht_samples_ns=11821,11430,10527,10124,9901,9897,9772,9667,9499,9401,9248,8995,9099,9997,10099,9048,9181,9215,9198,9114,9837,9038,9128,9251,9219,9789,9641,9037,8952,8861,9060,9713,8837,8934,8792,9578,9617,8944,8846,9476,8779,8716,8779,8848,8862
strata cell=occ1-offset-beyond-direct-absent-miss-wrapping-add occupancy=1 distribution=offset-beyond-direct operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=7c0c39940cd7f425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=47508 ht_median_ns=8972 ratio=5.2951 campaign_ratios=[5.2174,5.3404,5.3824] ratio_interval=[5.2174,5.3824] apeiron_samples_interval=[46176,49918] ht_samples_interval=[8461,9914] apeiron_ns_per_op=11.5986 ht_ns_per_op=2.1904 status=fail reason="ratio 5.2951 > 1.00"
strata cell=occ1-offset-beyond-direct-absent-miss-wrapping-add raw_apeiron_samples_ns=46700,47896,47062,47695,47353,47350,47588,47521,47694,47641,47874,47677,47335,46895,48645,46512,46176,47914,48127,48486,48032,48489,48360,48157,47883,49134,47273,47495,47043,47496,47283,47034,47367,47003,47197,47276,47414,47508,47229,47250,47311,49664,49918,49834,49742
strata cell=occ1-offset-beyond-direct-absent-miss-wrapping-add raw_ht_samples_ns=9121,8799,9418,9552,8839,9580,8814,9633,8896,9037,9251,9914,8915,9114,9773,9360,9530,9433,8967,9309,8800,8991,8753,8972,8689,8556,8666,9467,8793,8975,8827,8830,8973,8790,8790,8513,8604,9095,9181,8464,8553,9212,8489,8461,9205
strata cell=occ1-seeded-random-absent-miss-replacement occupancy=1 distribution=seeded-random operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=7eca0fa5fe53d425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=52097 ht_median_ns=9567 ratio=5.4455 campaign_ratios=[4.9214,5.5801,5.6555] ratio_interval=[4.9214,5.6555] apeiron_samples_interval=[51049,53871] ht_samples_interval=[8789,13527] apeiron_ns_per_op=12.7190 ht_ns_per_op=2.3357 status=fail reason="ratio 5.4455 > 1.00"
strata cell=occ1-seeded-random-absent-miss-replacement raw_apeiron_samples_ns=53871,52802,52529,52328,51968,52097,51943,52381,52196,52435,52013,52072,52334,52310,51712,52256,51866,51358,51332,51461,51427,51418,51049,53011,53258,52690,52363,52350,52279,52252,52649,52704,52747,52468,52100,51934,51976,51889,51757,51729,51669,51433,51354,51371,51311
strata cell=occ1-seeded-random-absent-miss-replacement raw_ht_samples_ns=13527,11952,11546,10771,10629,10908,10975,10527,9714,9567,9723,10170,9538,11067,10019,10183,10143,9454,9983,9180,9228,9227,10033,9147,9340,9398,9364,9353,10002,9295,9276,9834,9175,9065,9675,9756,8977,8966,9081,9585,8874,9543,8789,9464,8926
strata cell=occ1-seeded-random-absent-miss-wrapping-add occupancy=1 distribution=seeded-random operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=7eca0fa5fe53d425 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=51592 ht_median_ns=9008 ratio=5.7274 campaign_ratios=[5.7568,5.4444,5.8670] ratio_interval=[5.4444,5.8670] apeiron_samples_interval=[51262,53239] ht_samples_interval=[8659,9663] apeiron_ns_per_op=12.5957 ht_ns_per_op=2.1992 status=fail reason="ratio 5.7274 > 1.00"
strata cell=occ1-seeded-random-absent-miss-wrapping-add raw_apeiron_samples_ns=52466,51598,51546,51530,51581,51617,51615,51495,51427,51452,52992,52114,51729,51267,51378,51388,51363,51486,51316,51283,51526,51963,51775,51607,51713,51792,51864,51368,51423,51504,51262,53207,51770,51799,51900,51823,51913,51445,51472,51532,51688,51493,51592,51658,53239
strata cell=occ1-seeded-random-absent-miss-wrapping-add raw_ht_samples_ns=9098,9018,8960,9008,9001,8807,8903,8906,9580,8953,9500,9550,8918,8884,8868,9663,9479,8784,9493,9551,9460,9335,9414,8826,9515,8798,9497,9565,8827,9459,8810,9460,8789,8659,8727,9422,8799,9367,8672,9605,8708,9475,8843,9630,8719
strata cell=occ1-clustered-skewed-absent-miss-replacement occupancy=1 distribution=clustered-skewed operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=56672af0c80ac225 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=39353 ht_median_ns=38070 ratio=1.0337 campaign_ratios=[1.0605,1.0293,1.0298] ratio_interval=[1.0293,1.0605] apeiron_samples_interval=[38192,45658] ht_samples_interval=[36357,40771] apeiron_ns_per_op=9.6077 ht_ns_per_op=9.2944 status=fail reason="ratio 1.0337 > 1.00"
strata cell=occ1-clustered-skewed-absent-miss-replacement raw_apeiron_samples_ns=43798,43070,42294,42266,41418,45658,42324,40411,39842,39611,39359,39362,39199,39253,39024,38958,39353,38823,39595,39576,39550,39169,39910,39460,38961,38285,38809,38640,38739,39332,39218,39460,39229,38634,38192,38907,39272,39416,40335,39385,39158,39354,39226,38933,38547
strata cell=occ1-clustered-skewed-absent-miss-replacement raw_ht_samples_ns=37046,38622,38870,37110,38399,38688,37115,36573,37461,36555,38107,38129,38398,36557,38567,37137,36669,38581,36784,38054,37392,38445,36944,38582,40771,38075,36438,38070,36367,38610,37765,36357,40404,37665,38393,36962,38633,38092,37998,37176,38549,36906,38317,38720,38125
strata cell=occ1-clustered-skewed-absent-miss-wrapping-add occupancy=1 distribution=clustered-skewed operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=56672af0c80ac225 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=38943 ht_median_ns=37965 ratio=1.0258 campaign_ratios=[1.0244,1.0510,1.0293] ratio_interval=[1.0244,1.0510] apeiron_samples_interval=[37994,42487] ht_samples_interval=[36196,40763] apeiron_ns_per_op=9.5076 ht_ns_per_op=9.2688 status=fail reason="ratio 1.0258 > 1.00"
strata cell=occ1-clustered-skewed-absent-miss-wrapping-add raw_apeiron_samples_ns=42487,41077,40172,39626,39299,38912,38279,38465,38315,38319,38111,38958,38681,39147,38943,38777,39287,38810,39989,39642,40103,39788,39736,39745,38577,37994,38456,38284,38418,38339,38312,38392,38103,38722,38743,39017,39122,39702,39283,39521,39330,40386,39909,38839,38343
strata cell=occ1-clustered-skewed-absent-miss-wrapping-add raw_ht_samples_ns=37975,38720,37537,38167,37140,39633,38050,38195,36673,36415,36565,36519,38644,38016,38410,38366,36467,36632,36196,38271,36855,38461,36515,40763,38709,38205,36566,36925,36643,38389,36639,38057,38528,36294,38541,37907,38467,36607,37060,36944,37697,36664,37965,38850,38599
strata cell=occ2-direct-index-resident-absent-miss-replacement occupancy=2 distribution=direct-index-resident operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=7a4456e2b4f25725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10837 ht_median_ns=2181 ratio=4.9688 campaign_ratios=[4.9633,4.9711,4.9766] ratio_interval=[4.9633,4.9766] apeiron_samples_interval=[10820,10906] ht_samples_interval=[2175,3390] apeiron_ns_per_op=2.6458 ht_ns_per_op=0.5325 status=fail reason="ratio 4.9688 > 1.00"
strata cell=occ2-direct-index-resident-absent-miss-replacement raw_apeiron_samples_ns=10849,10893,10882,10899,10823,10883,10906,10834,10821,10830,10824,10825,10820,10826,10820,10832,10834,10833,10834,10830,10830,10831,10832,10831,10826,10830,10830,10837,10836,10838,10845,10840,10849,10844,10848,10845,10877,10847,10855,10849,10859,10855,10860,10860,10862
strata cell=occ2-direct-index-resident-absent-miss-replacement raw_ht_samples_ns=2244,2182,2184,2183,2183,2178,3390,2232,2181,2178,2243,2176,2179,2175,2180,2188,2179,2182,2177,2208,2178,2179,2177,2181,2182,2178,2181,2177,2182,2177,2175,2183,2175,2181,2177,2182,2175,2182,2179,2184,2177,2187,2180,2184,2179
strata cell=occ2-direct-index-resident-absent-miss-wrapping-add occupancy=2 distribution=direct-index-resident operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=7a4456e2b4f25725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10833 ht_median_ns=2180 ratio=4.9693 campaign_ratios=[4.9757,4.9670,4.9679] ratio_interval=[4.9670,4.9757] apeiron_samples_interval=[10820,12042] ht_samples_interval=[2174,2186] apeiron_ns_per_op=2.6448 ht_ns_per_op=0.5322 status=fail reason="ratio 4.9693 > 1.00"
strata cell=occ2-direct-index-resident-absent-miss-wrapping-add raw_apeiron_samples_ns=10858,10875,10852,10859,10847,10855,10848,10848,10842,10847,10838,10845,10833,10844,10832,10829,10832,10871,10835,10869,10827,10826,10828,10827,10825,10828,10839,10821,10824,10852,10832,10822,12042,10824,10830,10823,10830,10820,10830,10824,10839,10826,10836,10832,10841
strata cell=occ2-direct-index-resident-absent-miss-wrapping-add raw_ht_samples_ns=2180,2180,2186,2177,2185,2179,2184,2178,2186,2175,2186,2174,2183,2176,2182,2182,2176,2181,2176,2180,2178,2180,2178,2179,2181,2181,2183,2177,2182,2178,2174,2180,2177,2180,2178,2181,2178,2180,2177,2180,2175,2181,2175,2181,2180
strata cell=occ2-offset-beyond-direct-absent-miss-replacement occupancy=2 distribution=offset-beyond-direct operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=bae33451a8f25725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=42594 ht_median_ns=8270 ratio=5.1504 campaign_ratios=[5.0993,5.1068,5.1610] ratio_interval=[5.0993,5.1610] apeiron_samples_interval=[41831,44403] ht_samples_interval=[7344,10387] apeiron_ns_per_op=10.3989 ht_ns_per_op=2.0190 status=fail reason="ratio 5.1504 > 1.00"
strata cell=occ2-offset-beyond-direct-absent-miss-replacement raw_apeiron_samples_ns=44224,42911,43025,43135,42760,43016,42911,43037,42594,42633,42707,42547,42549,44056,43402,43686,43432,41896,41894,41941,42079,41952,43788,43776,44403,41935,42094,41870,41968,41947,44165,43991,44221,43553,44180,42273,41831,41903,41851,42046,41933,42472,42367,42401,42336
strata cell=occ2-offset-beyond-direct-absent-miss-replacement raw_ht_samples_ns=9356,8365,8756,8793,8566,8733,8272,7607,8456,7603,7735,7673,8457,7614,8415,8151,8204,8347,8368,8336,8197,8174,8218,8204,8351,8270,8428,8132,8377,8127,8586,8206,8292,8459,8558,10387,8216,7471,8209,8187,8386,7993,8146,7344,8082
strata cell=occ2-offset-beyond-direct-absent-miss-wrapping-add occupancy=2 distribution=offset-beyond-direct operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=bae33451a8f25725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=41976 ht_median_ns=8117 ratio=5.1714 campaign_ratios=[5.1618,5.1430,5.2576] ratio_interval=[5.1430,5.2576] apeiron_samples_interval=[41381,44314] ht_samples_interval=[7323,9846] apeiron_ns_per_op=10.2480 ht_ns_per_op=1.9817 status=fail reason="ratio 5.1714 > 1.00"
strata cell=occ2-offset-beyond-direct-absent-miss-wrapping-add raw_apeiron_samples_ns=41382,41584,41381,41507,41490,41566,41537,41760,41776,41723,41704,41656,43498,41891,41888,42328,42271,42210,42244,42272,42317,42147,42136,44314,42619,42239,42548,42286,43132,43122,42698,42441,42288,42278,43855,41831,41982,41950,41842,41976,41814,41923,41803,41869,41774
strata cell=occ2-offset-beyond-direct-absent-miss-wrapping-add raw_ht_samples_ns=8190,7451,8321,7407,8248,7406,8236,7456,7926,7461,8379,8094,8189,8070,8028,8184,7899,8215,7323,8142,8385,8132,7362,8264,8333,8284,8222,8424,8247,8273,9846,7457,8029,7417,8021,7409,8117,7421,8247,7483,7404,7420,8156,7979,8197
strata cell=occ2-seeded-random-absent-miss-replacement occupancy=2 distribution=seeded-random operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=0cdc10e1de7f9325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=46405 ht_median_ns=10216 ratio=4.5424 campaign_ratios=[4.3545,4.7300,4.5889] ratio_interval=[4.3545,4.7300] apeiron_samples_interval=[45857,48168] ht_samples_interval=[9291,13236] apeiron_ns_per_op=11.3293 ht_ns_per_op=2.4941 status=fail reason="ratio 4.5424 > 1.00"
strata cell=occ2-seeded-random-absent-miss-replacement raw_apeiron_samples_ns=46617,46536,46345,46411,46375,46378,46500,46261,46457,47809,46252,46169,46269,46056,46096,45897,45871,45857,46010,47061,46608,46609,46683,46804,46515,46560,46614,46500,46363,46405,46602,46362,46316,46729,46428,46263,46302,46288,46052,46079,46022,48168,46471,46562,47944
strata cell=occ2-seeded-random-absent-miss-replacement raw_ht_samples_ns=13236,12118,12763,11418,11163,11544,11270,10372,10650,10534,10465,10053,10181,10328,10287,10216,10086,10746,9830,9883,10442,9765,10325,9773,9482,9592,9449,9628,10378,9834,10103,10469,9866,10523,10838,10150,10085,9446,10749,9994,9691,10113,10565,9487,9291
strata cell=occ2-seeded-random-absent-miss-wrapping-add occupancy=2 distribution=seeded-random operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=0cdc10e1de7f9325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=46352 ht_median_ns=9353 ratio=4.9558 campaign_ratios=[4.9384,5.0928,4.9292] ratio_interval=[4.9292,5.0928] apeiron_samples_interval=[45849,48574] ht_samples_interval=[8784,10624] apeiron_ns_per_op=11.3164 ht_ns_per_op=2.2834 status=fail reason="ratio 4.9558 > 1.00"
strata cell=occ2-seeded-random-absent-miss-wrapping-add raw_apeiron_samples_ns=46584,46375,46500,46643,48574,46403,46241,46334,46186,46140,46253,46085,45885,45907,45849,47769,47974,47589,47748,47829,47555,47622,47805,47563,47602,47365,48175,46318,46293,46345,46261,46531,46352,46374,46154,46392,47644,46231,46087,46226,45965,45999,46072,46075,46102
strata cell=occ2-seeded-random-absent-miss-wrapping-add raw_ht_samples_ns=9366,10624,9463,9510,10204,9565,9592,9290,9944,9161,9297,9312,9308,9207,9351,9444,9964,9202,9730,9255,9036,9250,9347,9675,9863,9353,9116,9311,9786,9309,9378,10288,10472,9582,9477,10159,9185,9087,9095,9574,8920,9489,9071,8883,8784
strata cell=occ2-clustered-skewed-absent-miss-replacement occupancy=2 distribution=clustered-skewed operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=5fd6b3bae2065b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=52684 ht_median_ns=53038 ratio=0.9933 campaign_ratios=[1.0017,0.9775,0.9895] ratio_interval=[0.9775,1.0017] apeiron_samples_interval=[51709,60338] ht_samples_interval=[51991,56216] apeiron_ns_per_op=12.8623 ht_ns_per_op=12.9487 status=pass reason="ratio 0.9933 <= 1.00"
strata cell=occ2-clustered-skewed-absent-miss-replacement raw_apeiron_samples_ns=60238,59358,60338,58971,57106,53525,53288,52622,53353,53244,52364,59553,52377,52317,52347,52786,53755,52708,51741,52320,51997,51898,51709,52430,53189,53297,52802,52684,52496,52360,52398,51980,51847,52087,52040,52182,52110,52042,52653,57545,56075,55703,53635,52840,53937
strata cell=occ2-clustered-skewed-absent-miss-replacement raw_ht_samples_ns=54545,53991,53726,54478,56102,53529,53038,52773,53263,53002,52894,54646,52727,52447,52695,53706,56216,52943,53922,54225,52366,52426,52749,54270,51991,53937,52851,54655,52111,55025,53143,52432,53081,52891,54954,52801,52739,53049,52953,54222,55530,53012,52269,52334,52821
strata cell=occ2-clustered-skewed-absent-miss-wrapping-add occupancy=2 distribution=clustered-skewed operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=5fd6b3bae2065b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=53883 ht_median_ns=53339 ratio=1.0102 campaign_ratios=[1.0008,1.0125,1.0068] ratio_interval=[1.0008,1.0125] apeiron_samples_interval=[52127,63114] ht_samples_interval=[51721,55428] apeiron_ns_per_op=13.1550 ht_ns_per_op=13.0222 status=fail reason="ratio 1.0102 > 1.00"
strata cell=occ2-clustered-skewed-absent-miss-wrapping-add raw_apeiron_samples_ns=53285,53227,53021,53006,52580,52645,52636,52728,52846,57327,56473,54008,53494,53066,52921,54226,54108,53971,52286,53699,54803,54071,53811,55202,63114,55401,54115,53523,53078,53146,52243,54042,53883,53963,53931,53743,55305,54105,54361,54065,56785,55124,55162,52527,52127
strata cell=occ2-clustered-skewed-absent-miss-wrapping-add raw_ht_samples_ns=52584,52862,52321,51721,52981,52280,55319,53663,52682,53539,53803,53336,52822,54661,54250,53221,52907,53339,55288,53009,54597,53678,52917,53235,55379,55428,53715,53402,52540,55135,52906,52212,53130,53679,54203,53215,54097,53698,54394,52413,54717,53732,55015,52167,53318
strata cell=occ4-direct-index-resident-absent-miss-replacement occupancy=4 distribution=direct-index-resident operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=e10c5f8c8bef8725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10847 ht_median_ns=2181 ratio=4.9734 campaign_ratios=[4.9743,4.9734,4.9734] ratio_interval=[4.9734,4.9743] apeiron_samples_interval=[10822,13702] ht_samples_interval=[2173,2194] apeiron_ns_per_op=2.6482 ht_ns_per_op=0.5325 status=fail reason="ratio 4.9734 > 1.00"
strata cell=occ4-direct-index-resident-absent-miss-replacement raw_apeiron_samples_ns=10832,10858,10870,10849,13702,10840,10843,10870,10863,10837,10828,10846,10858,10864,10847,10889,10851,10850,10826,10829,10842,10858,10850,10837,10823,10826,10858,10854,10840,10827,10858,10847,10835,10822,10828,10853,10852,10901,10874,10828,10838,10855,10850,10826,10837
strata cell=occ4-direct-index-resident-absent-miss-replacement raw_ht_samples_ns=2184,2182,2180,2189,2181,2179,2184,2178,2175,2185,2180,2177,2179,2182,2187,2181,2183,2174,2181,2181,2177,2180,2183,2173,2178,2185,2179,2180,2186,2177,2181,2187,2177,2180,2183,2181,2181,2194,2182,2178,2182,2181,2177,2182,2176
strata cell=occ4-direct-index-resident-absent-miss-wrapping-add occupancy=4 distribution=direct-index-resident operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=e10c5f8c8bef8725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10843 ht_median_ns=2182 ratio=4.9693 campaign_ratios=[4.9675,4.9775,4.9661] ratio_interval=[4.9661,4.9775] apeiron_samples_interval=[10820,12525] ht_samples_interval=[2176,2229] apeiron_ns_per_op=2.6472 ht_ns_per_op=0.5327 status=fail reason="ratio 4.9693 > 1.00"
strata cell=occ4-direct-index-resident-absent-miss-wrapping-add raw_apeiron_samples_ns=10822,10880,10846,10843,10839,10820,10822,10846,10856,10837,10830,10822,10835,10911,12292,10831,10860,10861,10856,10843,10833,10837,10865,10869,10863,10834,10835,10841,10866,10863,10836,12069,10869,10846,10835,10839,10849,12525,10855,10823,10839,10836,10840,10855,10846
strata cell=occ4-direct-index-resident-absent-miss-wrapping-add raw_ht_samples_ns=2183,2176,2181,2187,2178,2192,2184,2176,2182,2184,2180,2212,2184,2178,2182,2184,2179,2181,2185,2180,2179,2185,2179,2178,2190,2187,2177,2188,2182,2177,2192,2178,2181,2184,2181,2178,2187,2182,2187,2189,2190,2176,2193,2229,2177
strata cell=occ4-offset-beyond-direct-absent-miss-replacement occupancy=4 distribution=offset-beyond-direct operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=a7c80f1f63ef8725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=42923 ht_median_ns=7301 ratio=5.8791 campaign_ratios=[5.8725,5.5621,5.8703] ratio_interval=[5.5621,5.8725] apeiron_samples_interval=[42295,44805] ht_samples_interval=[7010,9828] apeiron_ns_per_op=10.4792 ht_ns_per_op=1.7825 status=fail reason="ratio 5.8791 > 1.00"
strata cell=occ4-offset-beyond-direct-absent-miss-replacement raw_apeiron_samples_ns=43065,43058,43251,43187,42881,42867,42704,42871,43770,43568,43480,43468,43470,43378,43475,43295,42923,44527,44410,43010,42925,42902,42978,42928,42936,42903,42754,42537,43782,42546,42399,42336,42295,42832,42648,42896,42735,42737,43001,42780,42532,42440,42432,42530,44805
strata cell=occ4-offset-beyond-direct-absent-miss-replacement raw_ht_samples_ns=8541,8159,7365,8225,7860,7275,8064,7046,7144,7157,8000,7265,7938,7010,7229,8217,8226,8131,7262,7882,7176,7091,8034,7301,7138,7245,8148,7231,8069,7718,7219,7186,7159,9828,7851,7156,7120,7162,8030,7276,7957,7265,7237,7808,7844
strata cell=occ4-offset-beyond-direct-absent-miss-wrapping-add occupancy=4 distribution=offset-beyond-direct operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=a7c80f1f63ef8725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=43645 ht_median_ns=7911 ratio=5.5170 campaign_ratios=[5.4420,5.5294,5.5344] ratio_interval=[5.4420,5.5344] apeiron_samples_interval=[42439,45617] ht_samples_interval=[7103,8160] apeiron_ns_per_op=10.6555 ht_ns_per_op=1.9314 status=fail reason="ratio 5.5170 > 1.00"
strata cell=occ4-offset-beyond-direct-absent-miss-wrapping-add raw_apeiron_samples_ns=43561,43668,43887,43708,43553,43834,45138,42439,42801,42654,42572,42632,42497,42447,42562,43823,43706,43289,43184,43146,42978,43144,43441,43645,43638,43703,45617,43565,43713,43850,43667,44086,43822,43737,43532,44859,44015,43034,43142,42921,44402,44566,44504,44291,44590
strata cell=occ4-offset-beyond-direct-absent-miss-wrapping-add raw_ht_samples_ns=7210,8053,7829,8160,7310,7273,7946,8116,7918,7857,7865,8139,7161,8155,7171,7859,7181,7960,7165,7892,8052,7911,7129,8037,7535,7993,7162,7217,8159,8103,7966,7888,8066,8027,7836,7926,7837,7953,7168,8004,7103,8080,7991,8061,7867
strata cell=occ4-seeded-random-absent-miss-replacement occupancy=4 distribution=seeded-random operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=4945d52aaf241725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=47896 ht_median_ns=11496 ratio=4.1663 campaign_ratios=[4.0084,4.2297,4.2275] ratio_interval=[4.0084,4.2297] apeiron_samples_interval=[47163,50334] ht_samples_interval=[11009,13739] apeiron_ns_per_op=11.6934 ht_ns_per_op=2.8066 status=fail reason="ratio 4.1663 > 1.00"
strata cell=occ4-seeded-random-absent-miss-replacement raw_apeiron_samples_ns=49849,49455,49406,48771,48466,48520,50334,48037,47547,47561,47405,47723,47534,47769,47397,48630,48871,48936,48868,48989,48998,48845,48740,49285,48902,47896,47675,47628,47312,47360,47517,47614,47225,47309,47199,47624,47184,47313,47163,47525,49729,48677,48479,47967,47737
strata cell=occ4-seeded-random-absent-miss-replacement raw_ht_samples_ns=13739,13265,13715,12893,12721,12319,12256,11963,11984,11847,11776,11572,11459,11326,11496,11588,11548,11588,11385,11707,11355,11334,11233,11537,11889,11768,11550,11732,11297,11461,12010,11254,11401,11280,11259,11192,11415,11310,11242,11035,11189,11164,11030,11009,11200
strata cell=occ4-seeded-random-absent-miss-wrapping-add occupancy=4 distribution=seeded-random operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=4945d52aaf241725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=47844 ht_median_ns=11292 ratio=4.2370 campaign_ratios=[4.2748,4.2113,4.1789] ratio_interval=[4.1789,4.2748] apeiron_samples_interval=[47328,50487] ht_samples_interval=[10955,13875] apeiron_ns_per_op=11.6807 ht_ns_per_op=2.7568 status=fail reason="ratio 4.2370 > 1.00"
strata cell=occ4-seeded-random-absent-miss-wrapping-add raw_apeiron_samples_ns=47929,47865,47719,47844,47647,47899,47707,48672,48588,49645,50487,48552,48320,48008,47328,47542,47492,47481,47638,47754,47741,47563,47812,47803,48076,47979,47978,49721,48405,48323,48379,49239,48521,48554,48142,47772,47596,47567,47413,47432,47344,47694,47405,47407,49101
strata cell=occ4-seeded-random-absent-miss-wrapping-add raw_ht_samples_ns=11485,11212,11209,11037,11088,11024,11207,10995,10955,11690,11794,11418,11270,11262,11495,11657,11630,11471,11595,11649,11459,11351,11367,11272,11282,11066,11090,11131,11007,11164,11413,11765,12128,11805,13875,11677,11662,11541,11318,11256,11273,11187,11270,11057,11292
strata cell=occ4-clustered-skewed-absent-miss-replacement occupancy=4 distribution=clustered-skewed operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=d9a18114fe927525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=70916 ht_median_ns=55486 ratio=1.2781 campaign_ratios=[1.2961,1.2663,1.2568] ratio_interval=[1.2568,1.2961] apeiron_samples_interval=[68622,82336] ht_samples_interval=[54697,58815] apeiron_ns_per_op=17.3135 ht_ns_per_op=13.5464 status=fail reason="ratio 1.2781 > 1.00"
strata cell=occ4-clustered-skewed-absent-miss-replacement raw_apeiron_samples_ns=77509,74330,72715,71592,71131,73215,71212,71149,72930,72776,73515,69943,70059,69820,69834,74346,73035,72277,69965,69677,69573,68894,82336,70150,69543,70803,69572,71658,69066,68622,78903,68854,72119,69091,71437,71420,70916,72439,70036,71711,69585,69192,69316,69270,69404
strata cell=occ4-clustered-skewed-absent-miss-replacement raw_ht_samples_ns=58815,55540,55579,55765,55284,55121,55201,54985,55067,56775,55677,54782,55237,54807,55098,56559,57768,57199,55919,55253,54826,55058,55514,54805,55486,55847,55207,54763,55077,55161,55337,55830,57052,55220,56056,55916,56235,55426,55270,54697,55549,55725,55707,55905,55771
strata cell=occ4-clustered-skewed-absent-miss-wrapping-add occupancy=4 distribution=clustered-skewed operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=d9a18114fe927525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=71796 ht_median_ns=55565 ratio=1.2921 campaign_ratios=[1.2865,1.2843,1.2910] ratio_interval=[1.2843,1.2910] apeiron_samples_interval=[68779,76634] ht_samples_interval=[55006,58162] apeiron_ns_per_op=17.5283 ht_ns_per_op=13.5657 status=fail reason="ratio 1.2921 > 1.00"
strata cell=occ4-clustered-skewed-absent-miss-wrapping-add raw_apeiron_samples_ns=69202,71663,68916,68779,70607,70710,69469,71542,71376,71122,76634,72646,72977,73737,74439,74776,73873,73854,70467,72521,73758,72235,71179,69893,69593,70387,72053,69910,69291,69230,70436,70084,72135,71887,72141,75489,73830,72469,71796,71118,70834,72755,72668,72876,71833
strata cell=occ4-clustered-skewed-absent-miss-wrapping-add raw_ht_samples_ns=55818,55261,55274,55930,55577,55530,55210,55337,55479,55053,56904,56217,55937,55006,55233,56175,55239,58162,55053,55267,55421,55144,57048,56530,55300,57265,55361,56315,55038,55738,56421,55494,55459,55397,57588,56608,57382,55565,56205,57542,56125,55255,55497,55755,55875
strata cell=occ8-direct-index-resident-absent-miss-replacement occupancy=8 distribution=direct-index-resident operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=3fb4d114ab833b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10846 ht_median_ns=2182 ratio=4.9707 campaign_ratios=[4.9771,4.9638,4.9720] ratio_interval=[4.9638,4.9771] apeiron_samples_interval=[10820,14846] ht_samples_interval=[2174,2202] apeiron_ns_per_op=2.6479 ht_ns_per_op=0.5327 status=fail reason="ratio 4.9707 > 1.00"
strata cell=occ8-direct-index-resident-absent-miss-replacement raw_apeiron_samples_ns=10836,10855,10850,10833,10858,10839,10850,10860,10834,10855,14846,10845,10859,10843,10846,10871,10845,10841,10878,10822,10855,10829,10832,10850,10826,10860,10840,10841,12218,10820,10880,10856,10827,10844,10840,10827,10854,10830,10849,10865,10873,10854,10825,10839,10854
strata cell=occ8-direct-index-resident-absent-miss-replacement raw_ht_samples_ns=2183,2180,2184,2180,2180,2178,2175,2202,2185,2182,2193,2178,2178,2179,2181,2179,2174,2178,2179,2185,2189,2187,2184,2189,2192,2181,2197,2182,2178,2189,2180,2181,2187,2183,2183,2179,2179,2180,2178,2178,2182,2182,2186,2186,2184
strata cell=occ8-direct-index-resident-absent-miss-wrapping-add occupancy=8 distribution=direct-index-resident operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=3fb4d114ab833b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10855 ht_median_ns=2180 ratio=4.9794 campaign_ratios=[4.9789,4.9803,4.9775] ratio_interval=[4.9775,4.9803] apeiron_samples_interval=[10827,10915] ht_samples_interval=[2173,2195] apeiron_ns_per_op=2.6501 ht_ns_per_op=0.5322 status=fail reason="ratio 4.9794 > 1.00"
strata cell=occ8-direct-index-resident-absent-miss-wrapping-add raw_apeiron_samples_ns=10849,10847,10879,10857,10859,10879,10833,10859,10829,10846,10852,10829,10866,10840,10846,10857,10852,10846,10874,10859,10844,10843,10859,10855,10873,10857,10865,10841,10858,10830,10915,10861,10827,10896,10843,10829,10866,10867,10883,10895,10837,10877,10833,10848,10846
strata cell=occ8-direct-index-resident-absent-miss-wrapping-add raw_ht_samples_ns=2182,2179,2178,2175,2177,2177,2180,2183,2183,2186,2185,2182,2179,2173,2179,2178,2177,2182,2180,2181,2183,2184,2182,2180,2177,2173,2176,2176,2177,2183,2179,2179,2195,2185,2184,2183,2182,2177,2180,2176,2180,2181,2186,2185,2185
strata cell=occ8-offset-beyond-direct-absent-miss-replacement occupancy=8 distribution=offset-beyond-direct operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=d280fbbbc9833b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=46565 ht_median_ns=6628 ratio=7.0255 campaign_ratios=[7.0341,7.0425,6.4401] ratio_interval=[6.4401,7.0425] apeiron_samples_interval=[46177,51303] ht_samples_interval=[6380,7443] apeiron_ns_per_op=11.3684 ht_ns_per_op=1.6182 status=fail reason="ratio 7.0255 > 1.00"
strata cell=occ8-offset-beyond-direct-absent-miss-replacement raw_apeiron_samples_ns=46396,46346,46353,46336,46251,51303,48178,46600,46572,46611,46653,46580,46508,46707,46610,46593,46675,46542,46638,46643,46565,46647,46680,46501,46592,46586,46611,46397,46633,46593,46367,46476,46369,46481,46442,46541,46177,46327,46515,46644,46225,46279,46313,46865,46246
strata cell=occ8-offset-beyond-direct-absent-miss-replacement raw_ht_samples_ns=7174,6745,7420,6622,7402,6671,7258,6574,7409,6567,6597,6582,6606,6578,6595,6616,7192,6591,6625,7374,6380,6502,6498,6621,6572,7443,7333,6597,6565,6669,6582,7372,7375,6572,7117,7276,6519,7213,7403,7390,6628,7383,6605,6964,7200
strata cell=occ8-offset-beyond-direct-absent-miss-wrapping-add occupancy=8 distribution=offset-beyond-direct operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=d280fbbbc9833b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=46489 ht_median_ns=6640 ratio=7.0014 campaign_ratios=[7.0375,7.0619,6.6534] ratio_interval=[6.6534,7.0619] apeiron_samples_interval=[45913,53942] ht_samples_interval=[6464,8850] apeiron_ns_per_op=11.3499 ht_ns_per_op=1.6211 status=fail reason="ratio 7.0014 > 1.00"
strata cell=occ8-offset-beyond-direct-absent-miss-wrapping-add raw_apeiron_samples_ns=46337,50847,48648,46480,53942,46331,46231,46465,46363,46474,48138,45965,46011,46109,45913,48221,47948,46436,46460,46564,46556,46435,46489,46589,48040,46190,46194,46253,46320,46288,46284,46650,46850,46785,48040,47044,46962,46979,46919,46760,46607,46382,46412,46553,46591
strata cell=occ8-offset-beyond-direct-absent-miss-wrapping-add raw_ht_samples_ns=7353,6679,7243,7045,6514,6588,6553,6518,6542,6554,7206,6513,7220,6543,7343,7349,6579,6601,6537,6683,6464,6526,6563,6516,7193,7235,6542,7353,6546,7362,8850,6544,7469,7349,7335,7100,6638,6641,6532,7028,6640,6540,8380,7216,6589
strata cell=occ8-seeded-random-absent-miss-replacement occupancy=8 distribution=seeded-random operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=21ec2416bbefff25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=54393 ht_median_ns=15220 ratio=3.5738 campaign_ratios=[3.6469,3.6615,3.4718] ratio_interval=[3.4718,3.6615] apeiron_samples_interval=[52744,59132] ht_samples_interval=[14240,17569] apeiron_ns_per_op=13.2795 ht_ns_per_op=3.7158 status=fail reason="ratio 3.5738 > 1.00"
strata cell=occ8-seeded-random-absent-miss-replacement raw_apeiron_samples_ns=54932,54265,54116,56536,56067,55335,55087,54714,54426,54385,54212,54587,54361,54264,54590,53966,53742,56191,54261,53945,53963,53558,53395,53204,53288,53053,54393,55102,55457,55252,58150,59132,57612,55196,54604,54277,55906,53150,52744,55863,54552,53811,53669,54489,54296
strata cell=occ8-seeded-random-absent-miss-replacement raw_ht_samples_ns=16597,15913,15816,15514,17569,15859,15403,14968,14941,14596,14850,14431,14412,14240,14420,16616,15220,16968,15872,16133,14738,14619,14553,15143,14245,14324,14430,14329,14265,17113,15893,16725,15897,15182,17285,16361,15713,15453,15439,16416,15785,15097,15017,14629,14414
strata cell=occ8-seeded-random-absent-miss-wrapping-add occupancy=8 distribution=seeded-random operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=21ec2416bbefff25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=54142 ht_median_ns=14689 ratio=3.6859 campaign_ratios=[3.6722,3.5474,3.8197] ratio_interval=[3.5474,3.8197] apeiron_samples_interval=[52799,57091] ht_samples_interval=[14146,18369] apeiron_ns_per_op=13.2183 ht_ns_per_op=3.5862 status=fail reason="ratio 3.6859 > 1.00"
strata cell=occ8-seeded-random-absent-miss-wrapping-add raw_apeiron_samples_ns=57091,54626,53941,53585,53217,53786,55404,53931,54363,54290,54024,54097,53893,53068,52799,54150,56218,55041,54507,54497,53958,53910,53907,53715,53525,53034,53457,53477,53631,55162,54142,54028,54470,55535,55767,55591,55290,55481,55321,54007,53530,55574,55594,55299,55457
strata cell=occ8-seeded-random-absent-miss-wrapping-add raw_ht_samples_ns=18369,16500,16124,15625,14932,14689,15332,14519,14579,14471,14554,14370,16618,14260,14544,16686,16257,15754,15252,15517,14390,15376,14569,15432,14288,14569,14315,15197,14170,15082,16183,14940,16896,15115,15823,14618,14483,14237,14473,14153,15141,14193,14246,14146,14172
strata cell=occ8-clustered-skewed-absent-miss-replacement occupancy=8 distribution=clustered-skewed operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=6543c3a25d039d25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=82011 ht_median_ns=40210 ratio=2.0396 campaign_ratios=[2.0323,2.0265,2.0444] ratio_interval=[2.0265,2.0444] apeiron_samples_interval=[78414,89720] ht_samples_interval=[39315,41744] apeiron_ns_per_op=20.0222 ht_ns_per_op=9.8169 status=fail reason="ratio 2.0396 > 1.00"
strata cell=occ8-clustered-skewed-absent-miss-replacement raw_apeiron_samples_ns=82333,84288,83593,83590,81310,82953,83216,79810,80331,87075,84637,81181,81309,82620,79624,81522,84887,83013,83165,79843,81014,79152,81807,81964,81075,83273,81240,78414,80964,78985,80715,83960,82011,82458,82892,82621,82350,83402,82001,83212,82549,80975,79710,89720,79486
strata cell=occ8-clustered-skewed-absent-miss-replacement raw_ht_samples_ns=41085,41415,40834,40441,40091,39880,40184,40949,41744,40654,40210,40856,39813,41223,39909,40969,41401,40886,39949,40089,39722,39928,39456,40201,39947,40349,40159,40206,39315,39784,41075,41304,40808,40747,40424,39817,40138,41462,40959,40049,39955,40254,40333,39949,39711
strata cell=occ8-clustered-skewed-absent-miss-wrapping-add occupancy=8 distribution=clustered-skewed operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=6543c3a25d039d25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=82230 ht_median_ns=40374 ratio=2.0367 campaign_ratios=[2.0250,2.0525,2.0291] ratio_interval=[2.0250,2.0525] apeiron_samples_interval=[79361,89967] ht_samples_interval=[39752,44015] apeiron_ns_per_op=20.0757 ht_ns_per_op=9.8569 status=fail reason="ratio 2.0367 > 1.00"
strata cell=occ8-clustered-skewed-absent-miss-wrapping-add raw_apeiron_samples_ns=84892,81305,80957,80455,82245,79832,82531,84918,84320,81873,80934,84869,82453,85052,82409,86023,86639,82527,80176,81880,79521,79361,79692,82313,81102,82230,81632,83937,83912,85962,89967,83984,81657,83632,82778,80491,82721,85808,82142,79670,80951,80599,80280,80728,81905
strata cell=occ8-clustered-skewed-absent-miss-wrapping-add raw_ht_samples_ns=42239,40904,41029,40330,40242,42174,39997,41374,44015,40515,40696,41837,40374,40695,39920,41827,40601,41004,40678,40383,39808,40052,40367,40347,39752,40025,39941,40064,40054,39976,41918,42222,40275,40194,41627,40010,40047,41596,41068,40713,40365,40352,40020,39874,40420
strata cell=occ16-direct-index-resident-absent-miss-replacement occupancy=16 distribution=direct-index-resident operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=22d76b84a84f5725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10843 ht_median_ns=2185 ratio=4.9625 campaign_ratios=[4.9629,4.9570,4.9684] ratio_interval=[4.9570,4.9684] apeiron_samples_interval=[10821,10880] ht_samples_interval=[2178,2238] apeiron_ns_per_op=2.6472 ht_ns_per_op=0.5334 status=fail reason="ratio 4.9625 > 1.00"
strata cell=occ16-direct-index-resident-absent-miss-replacement raw_apeiron_samples_ns=10880,10839,10845,10821,10840,10832,10841,10829,10853,10843,10858,10852,10856,10858,10844,10855,10853,10830,10837,10828,10831,10834,10832,10842,10833,10870,10830,10854,10836,10856,10856,10851,10854,10854,10857,10857,10846,10854,10840,10845,10827,10832,10835,10830,10839
strata cell=occ16-direct-index-resident-absent-miss-replacement raw_ht_samples_ns=2192,2185,2211,2180,2186,2189,2209,2189,2185,2185,2183,2182,2178,2180,2184,2188,2180,2188,2180,2190,2183,2189,2185,2188,2184,2188,2190,2185,2186,2184,2181,2183,2179,2184,2187,2184,2183,2180,2190,2178,2190,2181,2191,2183,2238
strata cell=occ16-direct-index-resident-absent-miss-wrapping-add occupancy=16 distribution=direct-index-resident operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=22d76b84a84f5725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=10842 ht_median_ns=2185 ratio=4.9620 campaign_ratios=[4.9597,4.9656,4.9593] ratio_interval=[4.9593,4.9656] apeiron_samples_interval=[10823,11019] ht_samples_interval=[2174,2267] apeiron_ns_per_op=2.6470 ht_ns_per_op=0.5334 status=fail reason="ratio 4.9620 > 1.00"
strata cell=occ16-direct-index-resident-absent-miss-wrapping-add raw_apeiron_samples_ns=10873,10825,10834,10826,10836,10888,11019,10872,10838,10834,10854,10837,10858,10842,10858,10854,10857,10843,10852,10834,10841,10824,10832,10834,10832,10837,10823,10840,10866,10850,10854,10855,10846,10854,10853,10865,10845,10855,10833,10848,10827,10831,10825,10827,10840
strata cell=occ16-direct-index-resident-absent-miss-wrapping-add raw_ht_samples_ns=2190,2182,2190,2185,2188,2252,2189,2183,2267,2186,2184,2187,2182,2183,2181,2182,2179,2183,2174,2187,2177,2189,2180,2188,2182,2192,2186,2235,2185,2182,2181,2183,2179,2187,2188,2199,2183,2177,2188,2179,2189,2181,2190,2239,2194
strata cell=occ16-offset-beyond-direct-absent-miss-replacement occupancy=16 distribution=offset-beyond-direct operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=0e988344a84f5725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=49260 ht_median_ns=5449 ratio=9.0402 campaign_ratios=[9.0472,9.0356,9.0165] ratio_interval=[9.0165,9.0472] apeiron_samples_interval=[49011,53002] ht_samples_interval=[5343,6510] apeiron_ns_per_op=12.0264 ht_ns_per_op=1.3303 status=fail reason="ratio 9.0402 > 1.00"
strata cell=occ16-offset-beyond-direct-absent-miss-replacement raw_apeiron_samples_ns=49238,49244,49278,49403,49239,49309,49423,49326,49218,49316,49180,51569,49082,49187,49078,49594,49634,49092,49179,49335,49281,49260,49643,50476,51221,53002,49542,49329,49045,50152,50644,49148,49080,49011,49359,49433,49225,49152,49220,49149,49242,49183,49096,49185,51056
strata cell=occ16-offset-beyond-direct-absent-miss-replacement raw_ht_samples_ns=5557,5524,5495,5411,5435,5443,5696,5385,5448,5366,5444,5413,5419,5406,5529,5539,5343,5437,5449,5581,5698,5405,5483,5740,5718,5554,5418,5409,5437,6510,5484,5409,5437,5416,5470,5397,5409,5455,5679,5628,5655,5435,5449,5750,5513
strata cell=occ16-offset-beyond-direct-absent-miss-wrapping-add occupancy=16 distribution=offset-beyond-direct operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=0e988344a84f5725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=49257 ht_median_ns=5552 ratio=8.8719 campaign_ratios=[8.7587,9.1553,9.0585] ratio_interval=[8.7587,9.1553] apeiron_samples_interval=[48928,52953] ht_samples_interval=[5346,6028] apeiron_ns_per_op=12.0256 ht_ns_per_op=1.3555 status=fail reason="ratio 8.8719 > 1.00"
strata cell=occ16-offset-beyond-direct-absent-miss-wrapping-add raw_apeiron_samples_ns=52953,49228,49213,48928,49381,49215,49480,49259,49187,49313,49331,49048,49257,49528,49482,52021,52408,52344,52206,52682,49048,49096,48987,49055,49058,51748,52353,49813,49701,50006,49087,49171,49216,49144,49342,49169,49115,48936,49234,49141,49354,49215,49445,49235,51406
strata cell=occ16-offset-beyond-direct-absent-miss-wrapping-add raw_ht_samples_ns=5673,5368,5848,5695,5629,5422,5665,5388,5501,5624,5640,5433,5547,5629,5552,5577,5663,5423,5376,5669,5619,5642,5687,5385,5364,5399,5462,5380,5680,5391,5429,5355,5366,5643,6028,5433,5346,5632,5420,5611,5421,5703,5416,5631,5582
strata cell=occ16-seeded-random-absent-miss-replacement occupancy=16 distribution=seeded-random operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=9eee2cddca79cb25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=67220 ht_median_ns=18039 ratio=3.7264 campaign_ratios=[3.7655,3.6885,3.7371] ratio_interval=[3.6885,3.7655] apeiron_samples_interval=[65592,72213] ht_samples_interval=[17272,19471] apeiron_ns_per_op=16.4111 ht_ns_per_op=4.4041 status=fail reason="ratio 3.7264 > 1.00"
strata cell=occ16-seeded-random-absent-miss-replacement raw_apeiron_samples_ns=70991,67910,67478,66232,71087,68803,68346,67417,68208,70382,69141,67138,66937,67222,66936,69663,67094,67208,66622,67696,66543,66689,65900,65829,65592,66471,65861,66379,66037,65928,69345,67220,69685,68778,68128,72213,66456,67529,66610,66456,67840,66573,68785,68638,66663
strata cell=occ16-seeded-random-absent-miss-replacement raw_ht_samples_ns=19471,18870,19357,18035,19089,18039,17806,17891,18707,18642,17978,17743,17397,17590,17518,19156,18189,17697,17638,17565,17272,18813,18375,18021,17610,18557,18211,18559,17793,17311,19127,18553,18778,18153,17743,18146,17851,18506,17532,17347,18975,18338,17949,18354,17751
strata cell=occ16-seeded-random-absent-miss-wrapping-add occupancy=16 distribution=seeded-random operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=9eee2cddca79cb25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=67235 ht_median_ns=18211 ratio=3.6920 campaign_ratios=[3.6319,3.6560,3.7577] ratio_interval=[3.6319,3.7577] apeiron_samples_interval=[65857,76236] ht_samples_interval=[17238,20844] apeiron_ns_per_op=16.4148 ht_ns_per_op=4.4460 status=fail reason="ratio 3.6920 > 1.00"
strata cell=occ16-seeded-random-absent-miss-wrapping-add raw_apeiron_samples_ns=69374,76236,66750,66276,67704,66966,67394,66759,66249,67187,68277,66603,68873,65857,69723,69159,67268,69993,66379,66024,67227,66894,68261,67059,71295,67870,66578,67631,66373,66199,69062,67537,67235,70310,66608,68252,66849,67709,66470,66195,66213,66494,68378,69463,68817
strata cell=occ16-seeded-random-absent-miss-wrapping-add raw_ht_samples_ns=19001,20844,18805,18499,18809,18045,18729,18166,17928,17756,18564,18085,17801,17546,18998,18685,18362,18917,18390,18203,18235,17668,18081,18388,17839,18519,17958,18802,20476,18560,18884,18209,17923,18304,17797,18125,17758,18950,17727,17422,17238,17261,18751,18211,17973
strata cell=occ16-clustered-skewed-absent-miss-replacement occupancy=16 distribution=clustered-skewed operation=absent-miss merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=259c4b2d5e699f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=97893 ht_median_ns=48376 ratio=2.0236 campaign_ratios=[2.0395,2.0253,2.0056] ratio_interval=[2.0056,2.0395] apeiron_samples_interval=[94742,107710] ht_samples_interval=[46501,52096] apeiron_ns_per_op=23.8997 ht_ns_per_op=11.8105 status=fail reason="ratio 2.0236 > 1.00"
strata cell=occ16-clustered-skewed-absent-miss-replacement raw_apeiron_samples_ns=100139,96538,98104,98290,98196,97758,95689,95744,97890,99477,100321,107710,96663,100791,97893,100136,102021,101810,98843,95895,96415,98742,96228,97506,97976,97592,98276,94907,94742,98552,95330,98916,97492,97737,99885,101744,100018,99616,99370,96177,97606,96996,95866,97805,95204
strata cell=occ16-clustered-skewed-absent-miss-replacement raw_ht_samples_ns=48101,47587,49335,46550,49339,49469,48488,47716,48727,47558,47241,51100,48532,47825,47027,49188,49518,51023,48376,47569,46958,49681,48014,47525,49330,47497,49041,47335,46513,52096,49714,48130,47472,46598,49016,48905,50114,48115,47509,46501,48731,49891,48740,47491,49018
strata cell=occ16-clustered-skewed-absent-miss-wrapping-add occupancy=16 distribution=clustered-skewed operation=absent-miss merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=259c4b2d5e699f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=98609 ht_median_ns=48150 ratio=2.0480 campaign_ratios=[2.0438,2.0474,2.0577] ratio_interval=[2.0438,2.0577] apeiron_samples_interval=[95163,103906] ht_samples_interval=[46795,52410] apeiron_ns_per_op=24.0745 ht_ns_per_op=11.7554 status=fail reason="ratio 2.0480 > 1.00"
strata cell=occ16-clustered-skewed-absent-miss-wrapping-add raw_apeiron_samples_ns=98609,98019,98810,97669,96215,98265,97248,97148,100867,98310,96865,96115,95922,102096,96168,98336,99839,97570,103583,103906,101668,99702,99149,95163,98711,100930,96509,99563,101946,100510,99555,98708,99226,97846,97161,97570,99958,101617,103117,95381,99209,103207,97299,95976,98043
strata cell=occ16-clustered-skewed-absent-miss-wrapping-add raw_ht_samples_ns=50760,47591,47802,47179,48911,47355,47467,49682,47771,51539,47145,48610,46904,47789,47833,48752,47761,48361,47267,48798,48697,49483,49370,48149,49340,47020,49034,52410,48150,48170,51084,49115,48180,47524,47970,47237,48570,47880,48388,47330,46795,50752,47080,47302,48558
strata cell=occ1-direct-index-resident-existing-update-replacement occupancy=1 distribution=direct-index-resident operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=cda421dfa6689725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18101 ht_median_ns=7234 ratio=2.5022 campaign_ratios=[2.4998,2.5030,2.5059] ratio_interval=[2.4998,2.5059] apeiron_samples_interval=[18033,20239] ht_samples_interval=[6508,7278] apeiron_ns_per_op=4.4192 ht_ns_per_op=1.7661 status=fail reason="ratio 2.5022 > 1.00"
strata cell=occ1-direct-index-resident-existing-update-wrapping-add occupancy=1 distribution=direct-index-resident operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=210e0096c2570e25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18174 ht_median_ns=6762 ratio=2.6877 campaign_ratios=[2.6740,2.6878,2.7024] ratio_interval=[2.6740,2.7024] apeiron_samples_interval=[18075,18299] ht_samples_interval=[6474,7312] apeiron_ns_per_op=4.4370 ht_ns_per_op=1.6509 status=fail reason="ratio 2.6877 > 1.00"
strata cell=occ1-offset-beyond-direct-existing-update-replacement occupancy=1 distribution=offset-beyond-direct operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=cda421dfa6689725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=28507 ht_median_ns=7275 ratio=3.9185 campaign_ratios=[3.9116,3.9496,3.9163] ratio_interval=[3.9116,3.9496] apeiron_samples_interval=[28408,29072] ht_samples_interval=[7245,7352] apeiron_ns_per_op=6.9597 ht_ns_per_op=1.7761 status=fail reason="ratio 3.9185 > 1.00"
strata cell=occ1-offset-beyond-direct-existing-update-wrapping-add occupancy=1 distribution=offset-beyond-direct operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=6add6a52c4570e25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=28591 ht_median_ns=7278 ratio=3.9284 campaign_ratios=[3.9421,3.9202,3.9128] ratio_interval=[3.9128,3.9421] apeiron_samples_interval=[28389,28811] ht_samples_interval=[7239,7596] apeiron_ns_per_op=6.9802 ht_ns_per_op=1.7769 status=fail reason="ratio 3.9284 > 1.00"
strata cell=occ1-seeded-random-existing-update-replacement occupancy=1 distribution=seeded-random operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=cda421dfa6689725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=28670 ht_median_ns=8139 ratio=3.5225 campaign_ratios=[3.5230,3.5257,3.5070] ratio_interval=[3.5070,3.5257] apeiron_samples_interval=[28488,30078] ht_samples_interval=[8091,8266] apeiron_ns_per_op=6.9995 ht_ns_per_op=1.9871 status=fail reason="ratio 3.5225 > 1.00"
strata cell=occ1-seeded-random-existing-update-wrapping-add occupancy=1 distribution=seeded-random operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=1a1c40fa248b8625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=29038 ht_median_ns=8203 ratio=3.5399 campaign_ratios=[3.5135,3.5360,3.5522] ratio_interval=[3.5135,3.5522] apeiron_samples_interval=[28710,31030] ht_samples_interval=[8121,8536] apeiron_ns_per_op=7.0894 ht_ns_per_op=2.0027 status=fail reason="ratio 3.5399 > 1.00"
strata cell=occ1-clustered-skewed-existing-update-replacement occupancy=1 distribution=clustered-skewed operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=cda421dfa6689725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=32778 ht_median_ns=7264 ratio=4.5124 campaign_ratios=[4.5430,4.4840,4.4122] ratio_interval=[4.4122,4.5430] apeiron_samples_interval=[31926,34385] ht_samples_interval=[7225,8389] apeiron_ns_per_op=8.0024 ht_ns_per_op=1.7734 status=fail reason="ratio 4.5124 > 1.00"
strata cell=occ1-clustered-skewed-existing-update-wrapping-add occupancy=1 distribution=clustered-skewed operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=ed798e68f921ea25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=32625 ht_median_ns=7312 ratio=4.4618 campaign_ratios=[4.4618,4.4836,4.4152] ratio_interval=[4.4152,4.4836] apeiron_samples_interval=[31820,34149] ht_samples_interval=[7240,8756] apeiron_ns_per_op=7.9651 ht_ns_per_op=1.7852 status=fail reason="ratio 4.4618 > 1.00"
strata cell=occ2-direct-index-resident-existing-update-replacement occupancy=2 distribution=direct-index-resident operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=e2031546a2240b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18079 ht_median_ns=7188 ratio=2.5152 campaign_ratios=[2.5181,2.5138,2.5140] ratio_interval=[2.5138,2.5181] apeiron_samples_interval=[18047,18238] ht_samples_interval=[6517,8914] apeiron_ns_per_op=4.4138 ht_ns_per_op=1.7549 status=fail reason="ratio 2.5152 > 1.00"
strata cell=occ2-direct-index-resident-existing-update-wrapping-add occupancy=2 distribution=direct-index-resident operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=0b07bac0f8d2c525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18094 ht_median_ns=7183 ratio=2.5190 campaign_ratios=[2.5187,2.5179,2.5214] ratio_interval=[2.5179,2.5214] apeiron_samples_interval=[18049,19762] ht_samples_interval=[6279,7266] apeiron_ns_per_op=4.4175 ht_ns_per_op=1.7537 status=fail reason="ratio 2.5190 > 1.00"
strata cell=occ2-offset-beyond-direct-existing-update-replacement occupancy=2 distribution=offset-beyond-direct operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=e2031546a2240b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=30699 ht_median_ns=7279 ratio=4.2175 campaign_ratios=[4.1955,4.2211,4.2244] ratio_interval=[4.1955,4.2244] apeiron_samples_interval=[30428,32151] ht_samples_interval=[7245,12397] apeiron_ns_per_op=7.4949 ht_ns_per_op=1.7771 status=fail reason="ratio 4.2175 > 1.00"
strata cell=occ2-offset-beyond-direct-existing-update-wrapping-add occupancy=2 distribution=offset-beyond-direct operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=c841a0a050d2c525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=30964 ht_median_ns=7356 ratio=4.2094 campaign_ratios=[4.2169,4.1947,4.2122] ratio_interval=[4.1947,4.2169] apeiron_samples_interval=[30675,32482] ht_samples_interval=[7308,7404] apeiron_ns_per_op=7.5596 ht_ns_per_op=1.7959 status=fail reason="ratio 4.2094 > 1.00"
strata cell=occ2-seeded-random-existing-update-replacement occupancy=2 distribution=seeded-random operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=e2031546a2240b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=34049 ht_median_ns=9288 ratio=3.6659 campaign_ratios=[3.6561,3.6629,3.6696] ratio_interval=[3.6561,3.6696] apeiron_samples_interval=[33662,36985] ht_samples_interval=[9180,10693] apeiron_ns_per_op=8.3127 ht_ns_per_op=2.2676 status=fail reason="ratio 3.6659 > 1.00"
strata cell=occ2-seeded-random-existing-update-wrapping-add occupancy=2 distribution=seeded-random operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=419b485399404125 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=34291 ht_median_ns=9537 ratio=3.5956 campaign_ratios=[3.5974,3.6108,3.5846] ratio_interval=[3.5846,3.6108] apeiron_samples_interval=[33991,34659] ht_samples_interval=[9413,9684] apeiron_ns_per_op=8.3718 ht_ns_per_op=2.3284 status=fail reason="ratio 3.5956 > 1.00"
strata cell=occ2-clustered-skewed-existing-update-replacement occupancy=2 distribution=clustered-skewed operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=e2031546a2240b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=32566 ht_median_ns=46758 ratio=0.6965 campaign_ratios=[0.6936,0.6925,0.6990] ratio_interval=[0.6925,0.6990] apeiron_samples_interval=[32050,34912] ht_samples_interval=[46632,49790] apeiron_ns_per_op=7.9507 ht_ns_per_op=11.4155 status=pass reason="ratio 0.6965 <= 1.00"
strata cell=occ2-clustered-skewed-existing-update-wrapping-add occupancy=2 distribution=clustered-skewed operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=c336f582d7083925 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=33350 ht_median_ns=46791 ratio=0.7127 campaign_ratios=[0.7086,0.7132,0.7158] ratio_interval=[0.7086,0.7158] apeiron_samples_interval=[32685,33971] ht_samples_interval=[46685,48539] apeiron_ns_per_op=8.1421 ht_ns_per_op=11.4236 status=pass reason="ratio 0.7127 <= 1.00"
strata cell=occ4-direct-index-resident-existing-update-replacement occupancy=4 distribution=direct-index-resident operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=923c5979650eb325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18168 ht_median_ns=6941 ratio=2.6175 campaign_ratios=[2.6141,2.6175,2.6214] ratio_interval=[2.6141,2.6214] apeiron_samples_interval=[18111,18338] ht_samples_interval=[6517,7277] apeiron_ns_per_op=4.4355 ht_ns_per_op=1.6946 status=fail reason="ratio 2.6175 > 1.00"
strata cell=occ4-direct-index-resident-existing-update-wrapping-add occupancy=4 distribution=direct-index-resident operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=31f3abcacd58c725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18081 ht_median_ns=6693 ratio=2.7015 campaign_ratios=[2.6827,2.7180,2.6930] ratio_interval=[2.6827,2.7180] apeiron_samples_interval=[18035,19321] ht_samples_interval=[6339,7290] apeiron_ns_per_op=4.4143 ht_ns_per_op=1.6340 status=fail reason="ratio 2.7015 > 1.00"
strata cell=occ4-offset-beyond-direct-existing-update-replacement occupancy=4 distribution=offset-beyond-direct operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=923c5979650eb325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=30905 ht_median_ns=7472 ratio=4.1361 campaign_ratios=[4.1266,4.1392,4.1426] ratio_interval=[4.1266,4.1426] apeiron_samples_interval=[30708,31193] ht_samples_interval=[7426,7546] apeiron_ns_per_op=7.5452 ht_ns_per_op=1.8242 status=fail reason="ratio 4.1361 > 1.00"
strata cell=occ4-offset-beyond-direct-existing-update-wrapping-add occupancy=4 distribution=offset-beyond-direct operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=5b56cfd89158c725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=30949 ht_median_ns=7606 ratio=4.0690 campaign_ratios=[4.0649,4.0886,4.0532] ratio_interval=[4.0532,4.0886] apeiron_samples_interval=[30726,32652] ht_samples_interval=[7523,7699] apeiron_ns_per_op=7.5559 ht_ns_per_op=1.8569 status=fail reason="ratio 4.0690 > 1.00"
strata cell=occ4-seeded-random-existing-update-replacement occupancy=4 distribution=seeded-random operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=923c5979650eb325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=34704 ht_median_ns=9685 ratio=3.5833 campaign_ratios=[3.5670,3.5730,3.5918] ratio_interval=[3.5670,3.5918] apeiron_samples_interval=[34242,36511] ht_samples_interval=[9619,9873] apeiron_ns_per_op=8.4727 ht_ns_per_op=2.3645 status=fail reason="ratio 3.5833 > 1.00"
strata cell=occ4-seeded-random-existing-update-wrapping-add occupancy=4 distribution=seeded-random operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=762c16ec0111ff25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=34906 ht_median_ns=9857 ratio=3.5412 campaign_ratios=[3.5406,3.5217,3.5615] ratio_interval=[3.5217,3.5615] apeiron_samples_interval=[34350,49075] ht_samples_interval=[9674,10044] apeiron_ns_per_op=8.5220 ht_ns_per_op=2.4065 status=fail reason="ratio 3.5412 > 1.00"
strata cell=occ4-clustered-skewed-existing-update-replacement occupancy=4 distribution=clustered-skewed operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=923c5979650eb325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=42126 ht_median_ns=52581 ratio=0.8012 campaign_ratios=[0.8114,0.7982,0.7860] ratio_interval=[0.7860,0.8114] apeiron_samples_interval=[40085,47876] ht_samples_interval=[51934,56186] apeiron_ns_per_op=10.2847 ht_ns_per_op=12.8372 status=pass reason="ratio 0.8012 <= 1.00"
strata cell=occ4-clustered-skewed-existing-update-wrapping-add occupancy=4 distribution=clustered-skewed operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=1a1bed2fd3c9c725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=43187 ht_median_ns=52543 ratio=0.8219 campaign_ratios=[0.8169,0.8280,0.8219] ratio_interval=[0.8169,0.8280] apeiron_samples_interval=[41772,47112] ht_samples_interval=[51991,54208] apeiron_ns_per_op=10.5437 ht_ns_per_op=12.8279 status=pass reason="ratio 0.8219 <= 1.00"
strata cell=occ8-direct-index-resident-existing-update-replacement occupancy=8 distribution=direct-index-resident operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=a985f1ed867d0325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18200 ht_median_ns=6733 ratio=2.7031 campaign_ratios=[2.6955,2.7470,2.7014] ratio_interval=[2.6955,2.7470] apeiron_samples_interval=[18140,18425] ht_samples_interval=[6504,7225] apeiron_ns_per_op=4.4434 ht_ns_per_op=1.6438 status=fail reason="ratio 2.7031 > 1.00"
strata cell=occ8-direct-index-resident-existing-update-wrapping-add occupancy=8 distribution=direct-index-resident operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=0c294f38092d1925 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18291 ht_median_ns=6566 ratio=2.7857 campaign_ratios=[2.7615,2.8792,2.8150] ratio_interval=[2.7615,2.8792] apeiron_samples_interval=[18157,19951] ht_samples_interval=[6346,7343] apeiron_ns_per_op=4.4656 ht_ns_per_op=1.6030 status=fail reason="ratio 2.7857 > 1.00"
strata cell=occ8-offset-beyond-direct-existing-update-replacement occupancy=8 distribution=offset-beyond-direct operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=a985f1ed867d0325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=30111 ht_median_ns=7463 ratio=4.0347 campaign_ratios=[4.0041,4.0543,4.0220] ratio_interval=[4.0041,4.0543] apeiron_samples_interval=[29788,31632] ht_samples_interval=[7352,7624] apeiron_ns_per_op=7.3513 ht_ns_per_op=1.8220 status=fail reason="ratio 4.0347 > 1.00"
strata cell=occ8-offset-beyond-direct-existing-update-wrapping-add occupancy=8 distribution=offset-beyond-direct operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=31b86ad0232d1925 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=30089 ht_median_ns=7383 ratio=4.0754 campaign_ratios=[4.0822,4.0687,4.0831] ratio_interval=[4.0687,4.0831] apeiron_samples_interval=[29886,47065] ht_samples_interval=[7331,8037] apeiron_ns_per_op=7.3459 ht_ns_per_op=1.8025 status=fail reason="ratio 4.0754 > 1.00"
strata cell=occ8-seeded-random-existing-update-replacement occupancy=8 distribution=seeded-random operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=a985f1ed867d0325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=38514 ht_median_ns=10464 ratio=3.6806 campaign_ratios=[3.6902,3.7111,3.6064] ratio_interval=[3.6064,3.7111] apeiron_samples_interval=[37468,43110] ht_samples_interval=[9839,13212] apeiron_ns_per_op=9.4028 ht_ns_per_op=2.5547 status=fail reason="ratio 3.6806 > 1.00"
strata cell=occ8-seeded-random-existing-update-wrapping-add occupancy=8 distribution=seeded-random operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=37646a5013e70125 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=39243 ht_median_ns=10738 ratio=3.6546 campaign_ratios=[3.6230,3.6125,3.6580] ratio_interval=[3.6125,3.6580] apeiron_samples_interval=[38121,44032] ht_samples_interval=[10036,14379] apeiron_ns_per_op=9.5808 ht_ns_per_op=2.6216 status=fail reason="ratio 3.6546 > 1.00"
strata cell=occ8-clustered-skewed-existing-update-replacement occupancy=8 distribution=clustered-skewed operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=a985f1ed867d0325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=64204 ht_median_ns=56566 ratio=1.1350 campaign_ratios=[1.1446,1.1232,1.1251] ratio_interval=[1.1232,1.1446] apeiron_samples_interval=[60980,73738] ht_samples_interval=[55760,59446] apeiron_ns_per_op=15.6748 ht_ns_per_op=13.8101 status=fail reason="ratio 1.1350 > 1.00"
strata cell=occ8-clustered-skewed-existing-update-wrapping-add occupancy=8 distribution=clustered-skewed operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=5c9cffb9d678ed25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=65606 ht_median_ns=56535 ratio=1.1604 campaign_ratios=[1.1423,1.1543,1.1861] ratio_interval=[1.1423,1.1861] apeiron_samples_interval=[62398,71941] ht_samples_interval=[55745,59472] apeiron_ns_per_op=16.0171 ht_ns_per_op=13.8025 status=fail reason="ratio 1.1604 > 1.00"
strata cell=occ16-direct-index-resident-existing-update-replacement occupancy=16 distribution=direct-index-resident operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=0b645e7e27bb8325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18210 ht_median_ns=19236 ratio=0.9467 campaign_ratios=[0.9488,0.9455,0.9465] ratio_interval=[0.9455,0.9488] apeiron_samples_interval=[18134,19904] ht_samples_interval=[18888,19462] apeiron_ns_per_op=4.4458 ht_ns_per_op=4.6963 status=pass reason="ratio 0.9467 <= 1.00"
strata cell=occ16-direct-index-resident-existing-update-wrapping-add occupancy=16 distribution=direct-index-resident operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=2aad3afc8a060525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18200 ht_median_ns=18909 ratio=0.9625 campaign_ratios=[0.9671,0.9585,0.9693] ratio_interval=[0.9585,0.9693] apeiron_samples_interval=[18088,19438] ht_samples_interval=[18438,19256] apeiron_ns_per_op=4.4434 ht_ns_per_op=4.6165 status=pass reason="ratio 0.9625 <= 1.00"
strata cell=occ16-offset-beyond-direct-existing-update-replacement occupancy=16 distribution=offset-beyond-direct operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=0b645e7e27bb8325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=31248 ht_median_ns=7622 ratio=4.0997 campaign_ratios=[4.1233,4.0969,4.0898] ratio_interval=[4.0898,4.1233] apeiron_samples_interval=[31098,33057] ht_samples_interval=[7506,9105] apeiron_ns_per_op=7.6289 ht_ns_per_op=1.8608 status=fail reason="ratio 4.0997 > 1.00"
strata cell=occ16-offset-beyond-direct-existing-update-wrapping-add occupancy=16 distribution=offset-beyond-direct operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=c91bf0994a060525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=31527 ht_median_ns=7629 ratio=4.1325 campaign_ratios=[4.1493,4.1284,4.1282] ratio_interval=[4.1282,4.1493] apeiron_samples_interval=[31441,33852] ht_samples_interval=[7538,7817] apeiron_ns_per_op=7.6970 ht_ns_per_op=1.8625 status=fail reason="ratio 4.1325 > 1.00"
strata cell=occ16-seeded-random-existing-update-replacement occupancy=16 distribution=seeded-random operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=0b645e7e27bb8325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=49329 ht_median_ns=10857 ratio=4.5435 campaign_ratios=[4.5556,4.5154,4.5893] ratio_interval=[4.5154,4.5893] apeiron_samples_interval=[47169,55678] ht_samples_interval=[10420,11248] apeiron_ns_per_op=12.0432 ht_ns_per_op=2.6506 status=fail reason="ratio 4.5435 > 1.00"
strata cell=occ16-seeded-random-existing-update-wrapping-add occupancy=16 distribution=seeded-random operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=ea06716a52ec1125 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=48975 ht_median_ns=10899 ratio=4.4935 campaign_ratios=[4.5117,4.5022,4.4677] ratio_interval=[4.4677,4.5117] apeiron_samples_interval=[46774,53083] ht_samples_interval=[10524,12650] apeiron_ns_per_op=11.9568 ht_ns_per_op=2.6609 status=fail reason="ratio 4.4935 > 1.00"
strata cell=occ16-clustered-skewed-existing-update-replacement occupancy=16 distribution=clustered-skewed operation=existing-update merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=0b645e7e27bb8325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=78202 ht_median_ns=47323 ratio=1.6525 campaign_ratios=[1.6509,1.6501,1.6597] ratio_interval=[1.6501,1.6597] apeiron_samples_interval=[75570,81880] ht_samples_interval=[47201,51175] apeiron_ns_per_op=19.0923 ht_ns_per_op=11.5535 status=fail reason="ratio 1.6525 > 1.00"
strata cell=occ16-clustered-skewed-existing-update-wrapping-add occupancy=16 distribution=clustered-skewed operation=existing-update merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=1.000 oracle_digest=daa258e7ab742525 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=78064 ht_median_ns=47324 ratio=1.6496 campaign_ratios=[1.6364,1.6543,1.6521] ratio_interval=[1.6364,1.6543] apeiron_samples_interval=[75703,81972] ht_samples_interval=[47175,49280] apeiron_ns_per_op=19.0586 ht_ns_per_op=11.5537 status=fail reason="ratio 1.6496 > 1.00"
strata cell=occ1-direct-index-resident-admitted-fresh-insert-replacement occupancy=1 distribution=direct-index-resident operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=256 hit_ratio=0.000 oracle_digest=70578dc8edb71625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=1447 ht_median_ns=1045 ratio=1.3847 campaign_ratios=[1.3672,1.3990,1.3921] ratio_interval=[1.3672,1.3990] apeiron_samples_interval=[1414,3007] ht_samples_interval=[983,1205] apeiron_ns_per_op=5.6523 ht_ns_per_op=4.0820 status=fail reason="ratio 1.3847 > 1.00"
strata cell=occ1-direct-index-resident-admitted-fresh-insert-wrapping-add occupancy=1 distribution=direct-index-resident operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=256 hit_ratio=0.000 oracle_digest=70578dc8edb71625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=1451 ht_median_ns=984 ratio=1.4746 campaign_ratios=[1.4762,1.4836,1.4877] ratio_interval=[1.4762,1.4877] apeiron_samples_interval=[1409,1481] ht_samples_interval=[962,2585] apeiron_ns_per_op=5.6680 ht_ns_per_op=3.8438 status=fail reason="ratio 1.4746 > 1.00"
strata cell=occ1-offset-beyond-direct-admitted-fresh-insert-replacement occupancy=1 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=3840 hit_ratio=0.000 oracle_digest=c06028664b3ab625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=45872 ht_median_ns=25198 ratio=1.8205 campaign_ratios=[1.8307,1.8108,1.8263] ratio_interval=[1.8108,1.8307] apeiron_samples_interval=[45556,47658] ht_samples_interval=[24711,26938] apeiron_ns_per_op=11.9458 ht_ns_per_op=6.5620 status=fail reason="ratio 1.8205 > 1.00"
strata cell=occ1-offset-beyond-direct-admitted-fresh-insert-wrapping-add occupancy=1 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=3840 hit_ratio=0.000 oracle_digest=c06028664b3ab625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=45800 ht_median_ns=25148 ratio=1.8212 campaign_ratios=[1.8253,1.8034,1.8263] ratio_interval=[1.8034,1.8263] apeiron_samples_interval=[45370,54848] ht_samples_interval=[24861,70181] apeiron_ns_per_op=11.9271 ht_ns_per_op=6.5490 status=fail reason="ratio 1.8212 > 1.00"
strata cell=occ1-seeded-random-admitted-fresh-insert-replacement occupancy=1 distribution=seeded-random operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=3840 hit_ratio=0.000 oracle_digest=6237e261d08a4625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=54947 ht_median_ns=39082 ratio=1.4059 campaign_ratios=[1.4097,1.3867,1.4531] ratio_interval=[1.3867,1.4531] apeiron_samples_interval=[54047,59095] ht_samples_interval=[37052,48563] apeiron_ns_per_op=14.3091 ht_ns_per_op=10.1776 status=fail reason="ratio 1.4059 > 1.00"
strata cell=occ1-seeded-random-admitted-fresh-insert-wrapping-add occupancy=1 distribution=seeded-random operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=3840 hit_ratio=0.000 oracle_digest=6237e261d08a4625 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=3 apeiron_key_compares=MISSING apeiron_median_ns=54710 ht_median_ns=37739 ratio=1.4497 campaign_ratios=[1.4098,1.4799,1.4688] ratio_interval=[1.4098,1.4799] apeiron_samples_interval=[53879,61053] ht_samples_interval=[36330,43647] apeiron_ns_per_op=14.2474 ht_ns_per_op=9.8279 status=fail reason="ratio 1.4497 > 1.00"
strata cell=occ1-clustered-skewed-admitted-fresh-insert-replacement occupancy=1 distribution=clustered-skewed operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=3840 hit_ratio=0.000 oracle_digest=b3305ce3c6739825 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=85792 ht_median_ns=294075 ratio=0.2917 campaign_ratios=[0.2958,0.2903,0.2919] ratio_interval=[0.2903,0.2958] apeiron_samples_interval=[83657,101790] ht_samples_interval=[290556,299303] apeiron_ns_per_op=22.3417 ht_ns_per_op=76.5820 status=pass reason="ratio 0.2917 <= 1.00"
strata cell=occ1-clustered-skewed-admitted-fresh-insert-wrapping-add occupancy=1 distribution=clustered-skewed operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=256 simultaneous_population=true ops=3840 hit_ratio=0.000 oracle_digest=b3305ce3c6739825 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=256 ht_len=256 oracle_len=256 ht_capacity=512 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=86889 ht_median_ns=292962 ratio=0.2966 campaign_ratios=[0.2990,0.2942,0.2956] ratio_interval=[0.2942,0.2990] apeiron_samples_interval=[83878,90452] ht_samples_interval=[290298,300693] apeiron_ns_per_op=22.6273 ht_ns_per_op=76.2922 status=pass reason="ratio 0.2966 <= 1.00"
strata cell=occ2-direct-index-resident-admitted-fresh-insert-replacement occupancy=2 distribution=direct-index-resident operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=512 hit_ratio=0.000 oracle_digest=29eb332515bfa325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=2437 ht_median_ns=1982 ratio=1.2296 campaign_ratios=[1.2326,1.2288,1.2292] ratio_interval=[1.2288,1.2326] apeiron_samples_interval=[2418,2496] ht_samples_interval=[1939,2096] apeiron_ns_per_op=4.7598 ht_ns_per_op=3.8711 status=fail reason="ratio 1.2296 > 1.00"
strata cell=occ2-direct-index-resident-admitted-fresh-insert-wrapping-add occupancy=2 distribution=direct-index-resident operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=512 hit_ratio=0.000 oracle_digest=29eb332515bfa325 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=2411 ht_median_ns=1948 ratio=1.2377 campaign_ratios=[1.2393,1.2369,1.2382] ratio_interval=[1.2369,1.2393] apeiron_samples_interval=[2396,2435] ht_samples_interval=[1895,2012] apeiron_ns_per_op=4.7090 ht_ns_per_op=3.8047 status=fail reason="ratio 1.2377 > 1.00"
strata cell=occ2-offset-beyond-direct-admitted-fresh-insert-replacement occupancy=2 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=3584 hit_ratio=0.000 oracle_digest=8d647c649117bb25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=42655 ht_median_ns=23494 ratio=1.8156 campaign_ratios=[1.8138,1.8138,1.8160] ratio_interval=[1.8138,1.8160] apeiron_samples_interval=[42363,44509] ht_samples_interval=[23273,25019] apeiron_ns_per_op=11.9015 ht_ns_per_op=6.5552 status=fail reason="ratio 1.8156 > 1.00"
strata cell=occ2-offset-beyond-direct-admitted-fresh-insert-wrapping-add occupancy=2 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=3584 hit_ratio=0.000 oracle_digest=8d647c649117bb25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=42778 ht_median_ns=23457 ratio=1.8237 campaign_ratios=[1.8186,1.8252,1.8191] ratio_interval=[1.8186,1.8252] apeiron_samples_interval=[42451,45005] ht_samples_interval=[23212,26589] apeiron_ns_per_op=11.9358 ht_ns_per_op=6.5449 status=fail reason="ratio 1.8237 > 1.00"
strata cell=occ2-seeded-random-admitted-fresh-insert-replacement occupancy=2 distribution=seeded-random operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=3584 hit_ratio=0.000 oracle_digest=1e0f1a043ebb0725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=52922 ht_median_ns=38663 ratio=1.3688 campaign_ratios=[1.3459,1.3849,1.3704] ratio_interval=[1.3459,1.3849] apeiron_samples_interval=[52309,59004] ht_samples_interval=[36988,52383] apeiron_ns_per_op=14.7662 ht_ns_per_op=10.7877 status=fail reason="ratio 1.3688 > 1.00"
strata cell=occ2-seeded-random-admitted-fresh-insert-wrapping-add occupancy=2 distribution=seeded-random operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=3584 hit_ratio=0.000 oracle_digest=1e0f1a043ebb0725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=5 apeiron_key_compares=MISSING apeiron_median_ns=52874 ht_median_ns=38531 ratio=1.3722 campaign_ratios=[1.3333,1.3734,1.3796] ratio_interval=[1.3333,1.3796] apeiron_samples_interval=[51863,56661] ht_samples_interval=[37431,45561] apeiron_ns_per_op=14.7528 ht_ns_per_op=10.7508 status=fail reason="ratio 1.3722 > 1.00"
strata cell=occ2-clustered-skewed-admitted-fresh-insert-replacement occupancy=2 distribution=clustered-skewed operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=3584 hit_ratio=0.000 oracle_digest=67f916c3c30a6725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=80030 ht_median_ns=271908 ratio=0.2943 campaign_ratios=[0.2945,0.2980,0.2917] ratio_interval=[0.2917,0.2980] apeiron_samples_interval=[76976,83734] ht_samples_interval=[269356,277441] apeiron_ns_per_op=22.3298 ht_ns_per_op=75.8672 status=pass reason="ratio 0.2943 <= 1.00"
strata cell=occ2-clustered-skewed-admitted-fresh-insert-wrapping-add occupancy=2 distribution=clustered-skewed operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=512 simultaneous_population=true ops=3584 hit_ratio=0.000 oracle_digest=67f916c3c30a6725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=512 ht_len=512 oracle_len=512 ht_capacity=1024 ht_longest_probe=12 apeiron_key_compares=MISSING apeiron_median_ns=79309 ht_median_ns=270821 ratio=0.2928 campaign_ratios=[0.2971,0.2926,0.2894] ratio_interval=[0.2894,0.2971] apeiron_samples_interval=[77190,85501] ht_samples_interval=[268420,275301] apeiron_ns_per_op=22.1286 ht_ns_per_op=75.5639 status=pass reason="ratio 0.2928 <= 1.00"
strata cell=occ4-direct-index-resident-admitted-fresh-insert-replacement occupancy=4 distribution=direct-index-resident operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=1024 hit_ratio=0.000 oracle_digest=159dd553fc677f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=4762 ht_median_ns=4085 ratio=1.1657 campaign_ratios=[1.1728,1.1616,1.1655] ratio_interval=[1.1616,1.1728] apeiron_samples_interval=[4747,4790] ht_samples_interval=[3965,5226] apeiron_ns_per_op=4.6504 ht_ns_per_op=3.9893 status=fail reason="ratio 1.1657 > 1.00"
strata cell=occ4-direct-index-resident-admitted-fresh-insert-wrapping-add occupancy=4 distribution=direct-index-resident operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=1024 hit_ratio=0.000 oracle_digest=159dd553fc677f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=4759 ht_median_ns=4789 ratio=0.9937 campaign_ratios=[0.9912,0.9935,0.9956] ratio_interval=[0.9912,0.9956] apeiron_samples_interval=[4732,4791] ht_samples_interval=[4138,5876] apeiron_ns_per_op=4.6475 ht_ns_per_op=4.6768 status=pass reason="ratio 0.9937 <= 1.00"
strata cell=occ4-offset-beyond-direct-admitted-fresh-insert-replacement occupancy=4 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=3072 hit_ratio=0.000 oracle_digest=f9c869609a43ff25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=37839 ht_median_ns=20305 ratio=1.8635 campaign_ratios=[1.8701,1.8536,1.8596] ratio_interval=[1.8536,1.8701] apeiron_samples_interval=[37638,39486] ht_samples_interval=[19995,21792] apeiron_ns_per_op=12.3174 ht_ns_per_op=6.6097 status=fail reason="ratio 1.8635 > 1.00"
strata cell=occ4-offset-beyond-direct-admitted-fresh-insert-wrapping-add occupancy=4 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=3072 hit_ratio=0.000 oracle_digest=f9c869609a43ff25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=2 apeiron_key_compares=MISSING apeiron_median_ns=38661 ht_median_ns=20357 ratio=1.8992 campaign_ratios=[1.9016,1.9049,1.8623] ratio_interval=[1.8623,1.9049] apeiron_samples_interval=[37678,40166] ht_samples_interval=[20113,21517] apeiron_ns_per_op=12.5850 ht_ns_per_op=6.6266 status=fail reason="ratio 1.8992 > 1.00"
strata cell=occ4-seeded-random-admitted-fresh-insert-replacement occupancy=4 distribution=seeded-random operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=3072 hit_ratio=0.000 oracle_digest=02111fa20d83af25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=48006 ht_median_ns=35948 ratio=1.3354 campaign_ratios=[1.3492,1.3214,1.3505] ratio_interval=[1.3214,1.3505] apeiron_samples_interval=[47566,53962] ht_samples_interval=[35026,46037] apeiron_ns_per_op=15.6270 ht_ns_per_op=11.7018 status=fail reason="ratio 1.3354 > 1.00"
strata cell=occ4-seeded-random-admitted-fresh-insert-wrapping-add occupancy=4 distribution=seeded-random operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=3072 hit_ratio=0.000 oracle_digest=02111fa20d83af25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=48629 ht_median_ns=35981 ratio=1.3515 campaign_ratios=[1.3473,1.3710,1.3413] ratio_interval=[1.3413,1.3710] apeiron_samples_interval=[47500,52833] ht_samples_interval=[34234,44935] apeiron_ns_per_op=15.8298 ht_ns_per_op=11.7126 status=fail reason="ratio 1.3515 > 1.00"
strata cell=occ4-clustered-skewed-admitted-fresh-insert-replacement occupancy=4 distribution=clustered-skewed operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=3072 hit_ratio=0.000 oracle_digest=95e36dcf160c3d25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=69404 ht_median_ns=184226 ratio=0.3767 campaign_ratios=[0.3775,0.3756,0.3761] ratio_interval=[0.3756,0.3775] apeiron_samples_interval=[67202,72086] ht_samples_interval=[181694,229357] apeiron_ns_per_op=22.5924 ht_ns_per_op=59.9694 status=pass reason="ratio 0.3767 <= 1.00"
strata cell=occ4-clustered-skewed-admitted-fresh-insert-wrapping-add occupancy=4 distribution=clustered-skewed operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=1024 simultaneous_population=true ops=3072 hit_ratio=0.000 oracle_digest=95e36dcf160c3d25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=1024 ht_len=1024 oracle_len=1024 ht_capacity=2048 ht_longest_probe=191 apeiron_key_compares=MISSING apeiron_median_ns=70103 ht_median_ns=183391 ratio=0.3823 campaign_ratios=[0.3841,0.3871,0.3773] ratio_interval=[0.3773,0.3871] apeiron_samples_interval=[67784,73677] ht_samples_interval=[181722,190443] apeiron_ns_per_op=22.8200 ht_ns_per_op=59.6976 status=pass reason="ratio 0.3823 <= 1.00"
strata cell=occ8-direct-index-resident-admitted-fresh-insert-replacement occupancy=8 distribution=direct-index-resident operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=9e2f25698fc95b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=9535 ht_median_ns=10527 ratio=0.9058 campaign_ratios=[0.9043,0.9138,0.9040] ratio_interval=[0.9040,0.9138] apeiron_samples_interval=[9486,9584] ht_samples_interval=[10046,11005] apeiron_ns_per_op=4.6558 ht_ns_per_op=5.1401 status=pass reason="ratio 0.9058 <= 1.00"
strata cell=occ8-direct-index-resident-admitted-fresh-insert-wrapping-add occupancy=8 distribution=direct-index-resident operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=9e2f25698fc95b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=9530 ht_median_ns=10646 ratio=0.8952 campaign_ratios=[0.8831,0.9037,0.8952] ratio_interval=[0.8831,0.9037] apeiron_samples_interval=[9455,9606] ht_samples_interval=[10239,11917] apeiron_ns_per_op=4.6533 ht_ns_per_op=5.1982 status=pass reason="ratio 0.8952 <= 1.00"
strata cell=occ8-offset-beyond-direct-admitted-fresh-insert-replacement occupancy=8 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=3fd19b4a4fc95b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=25078 ht_median_ns=13466 ratio=1.8623 campaign_ratios=[1.8691,1.8574,1.8655] ratio_interval=[1.8574,1.8691] apeiron_samples_interval=[24907,25382] ht_samples_interval=[13247,15201] apeiron_ns_per_op=12.2451 ht_ns_per_op=6.5752 status=fail reason="ratio 1.8623 > 1.00"
strata cell=occ8-offset-beyond-direct-admitted-fresh-insert-wrapping-add occupancy=8 distribution=offset-beyond-direct operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=3fd19b4a4fc95b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=25218 ht_median_ns=13429 ratio=1.8779 campaign_ratios=[1.8695,1.8823,1.8784] ratio_interval=[1.8695,1.8823] apeiron_samples_interval=[24722,27495] ht_samples_interval=[13289,13663] apeiron_ns_per_op=12.3135 ht_ns_per_op=6.5571 status=fail reason="ratio 1.8779 > 1.00"
strata cell=occ8-seeded-random-admitted-fresh-insert-replacement occupancy=8 distribution=seeded-random operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=0e4310e8077e9f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=37262 ht_median_ns=23761 ratio=1.5682 campaign_ratios=[1.5590,1.5934,1.5829] ratio_interval=[1.5590,1.5934] apeiron_samples_interval=[35754,42115] ht_samples_interval=[22175,29930] apeiron_ns_per_op=18.1943 ht_ns_per_op=11.6021 status=fail reason="ratio 1.5682 > 1.00"
strata cell=occ8-seeded-random-admitted-fresh-insert-wrapping-add occupancy=8 distribution=seeded-random operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=0e4310e8077e9f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=9 apeiron_key_compares=MISSING apeiron_median_ns=36386 ht_median_ns=22788 ratio=1.5967 campaign_ratios=[1.5948,1.5880,1.5926] ratio_interval=[1.5880,1.5948] apeiron_samples_interval=[35603,40332] ht_samples_interval=[21981,27185] apeiron_ns_per_op=17.7666 ht_ns_per_op=11.1270 status=fail reason="ratio 1.5967 > 1.00"
strata cell=occ8-clustered-skewed-admitted-fresh-insert-replacement occupancy=8 distribution=clustered-skewed operation=admitted-fresh-insert merge=replacement seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=122905e1bccb3d25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=48292 ht_median_ns=107498 ratio=0.4492 campaign_ratios=[0.4501,0.4501,0.4462] ratio_interval=[0.4462,0.4501] apeiron_samples_interval=[47027,50800] ht_samples_interval=[106164,111744] apeiron_ns_per_op=23.5801 ht_ns_per_op=52.4893 status=pass reason="ratio 0.4492 <= 1.00"
strata cell=occ8-clustered-skewed-admitted-fresh-insert-wrapping-add occupancy=8 distribution=clustered-skewed operation=admitted-fresh-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=2048 simultaneous_population=true ops=2048 hit_ratio=0.000 oracle_digest=122905e1bccb3d25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=2048 ht_len=2048 oracle_len=2048 ht_capacity=4096 ht_longest_probe=170 apeiron_key_compares=MISSING apeiron_median_ns=48325 ht_median_ns=107894 ratio=0.4479 campaign_ratios=[0.4501,0.4483,0.4419] ratio_interval=[0.4419,0.4501] apeiron_samples_interval=[47206,51929] ht_samples_interval=[106329,111368] apeiron_ns_per_op=23.5962 ht_ns_per_op=52.6826 status=pass reason="ratio 0.4479 <= 1.00"
strata cell=occ16-direct-index-resident-growth-insert-replacement occupancy=16 distribution=direct-index-resident operation=growth-insert merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=5035b5a9c61b9725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=19030 ht_median_ns=32096 ratio=0.5929 campaign_ratios=[0.5926,0.5934,0.5929] ratio_interval=[0.5926,0.5934] apeiron_samples_interval=[18914,19124] ht_samples_interval=[31832,33619] apeiron_ns_per_op=4.6460 ht_ns_per_op=7.8359 status=pass reason="ratio 0.5929 <= 1.00"
strata cell=occ16-direct-index-resident-growth-insert-wrapping-add occupancy=16 distribution=direct-index-resident operation=growth-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=5035b5a9c61b9725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=18977 ht_median_ns=32076 ratio=0.5916 campaign_ratios=[0.5909,0.5910,0.5916] ratio_interval=[0.5909,0.5916] apeiron_samples_interval=[18895,21442] ht_samples_interval=[31912,33394] apeiron_ns_per_op=4.6331 ht_ns_per_op=7.8311 status=pass reason="ratio 0.5916 <= 1.00"
strata cell=occ16-offset-beyond-direct-growth-insert-replacement occupancy=16 distribution=offset-beyond-direct operation=growth-insert merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=eed487a5401b9725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=60724 ht_median_ns=27680 ratio=2.1938 campaign_ratios=[2.1905,2.1904,2.1988] ratio_interval=[2.1904,2.1988] apeiron_samples_interval=[59963,66464] ht_samples_interval=[27432,28044] apeiron_ns_per_op=14.8252 ht_ns_per_op=6.7578 status=fail reason="ratio 2.1938 > 1.00"
strata cell=occ16-offset-beyond-direct-growth-insert-wrapping-add occupancy=16 distribution=offset-beyond-direct operation=growth-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=eed487a5401b9725 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=1 apeiron_key_compares=MISSING apeiron_median_ns=60894 ht_median_ns=27654 ratio=2.2020 campaign_ratios=[2.2024,2.2025,2.2024] ratio_interval=[2.2024,2.2025] apeiron_samples_interval=[60500,65249] ht_samples_interval=[27418,28855] apeiron_ns_per_op=14.8667 ht_ns_per_op=6.7515 status=fail reason="ratio 2.2020 > 1.00"
strata cell=occ16-seeded-random-growth-insert-replacement occupancy=16 distribution=seeded-random operation=growth-insert merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=1899f740eae78b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=100534 ht_median_ns=69030 ratio=1.4564 campaign_ratios=[1.4757,1.4434,1.4655] ratio_interval=[1.4434,1.4757] apeiron_samples_interval=[95547,107850] ht_samples_interval=[64742,88549] apeiron_ns_per_op=24.5444 ht_ns_per_op=16.8530 status=fail reason="ratio 1.4564 > 1.00"
strata cell=occ16-seeded-random-growth-insert-wrapping-add occupancy=16 distribution=seeded-random operation=growth-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=1899f740eae78b25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=7 apeiron_key_compares=MISSING apeiron_median_ns=101464 ht_median_ns=68874 ratio=1.4732 campaign_ratios=[1.4783,1.4520,1.4797] ratio_interval=[1.4520,1.4797] apeiron_samples_interval=[96016,106137] ht_samples_interval=[65300,76786] apeiron_ns_per_op=24.7715 ht_ns_per_op=16.8149 status=fail reason="ratio 1.4732 > 1.00"
strata cell=occ16-clustered-skewed-growth-insert-replacement occupancy=16 distribution=clustered-skewed operation=growth-insert merge=replacement seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=226571a3ea055f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=215673 ht_median_ns=167838 ratio=1.2850 campaign_ratios=[1.2878,1.2723,1.2876] ratio_interval=[1.2723,1.2878] apeiron_samples_interval=[208286,224787] ht_samples_interval=[166024,226184] apeiron_ns_per_op=52.6545 ht_ns_per_op=40.9761 status=fail reason="ratio 1.2850 > 1.00"
strata cell=occ16-clustered-skewed-growth-insert-wrapping-add occupancy=16 distribution=clustered-skewed operation=growth-insert merge=wrapping-add seed=0x5eed1234abcd0001 population=4096 simultaneous_population=true ops=4096 hit_ratio=0.000 oracle_digest=226571a3ea055f25 apeiron_digest_ok=true ht_digest_ok=true apeiron_admitted=4096 ht_len=4096 oracle_len=4096 ht_capacity=8192 ht_longest_probe=13 apeiron_key_compares=MISSING apeiron_median_ns=216411 ht_median_ns=168130 ratio=1.2872 campaign_ratios=[1.2846,1.2831,1.2899] ratio_interval=[1.2831,1.2899] apeiron_samples_interval=[209241,349083] ht_samples_interval=[165682,173657] apeiron_ns_per_op=52.8347 ht_ns_per_op=41.0474 status=fail reason="ratio 1.2872 > 1.00"
strata_resource cell=zero-group-width-sizeoverflow operation=zero-group-width status=not-applicable population=0 ops=1 observed="SizeOverflow=true unchanged-state=true no-clone=true no-merge=true" reason="the N=0 contract refuses every write, keeps reads total and invokes no callback"
strata_resource cell=sizing-overflow-not-forceable operation=sizing-overflow status=missing population=0 ops=0 observed="not forceable: checked_mul(2) cannot overflow a real region length" reason="requires a region length above usize::MAX/2; registered missing, not fabricated"
strata_resource cell=injected-reservation-failure-not-forceable operation=injected-reservation-failure status=missing population=0 ops=0 observed="not forceable: try_reserve_exact failure needs an allocator shim" reason="no allocator injection point in this harness; registered missing, not fabricated"
strata_summary cells=160 measured=160 pass=20 fail=140 not_applicable=0 missing=0 rejected=0 resource_cells=3 resource_not_applicable=1 resource_missing=2 campaigns=3 warmup=2 samples=15 ops=4096 ht_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc="rustc 1.97.0-nightly (17584a181 2026-04-13)" cpu="Intel(R) Core(TM) i9-14900KF"
strata_family dimension=distribution value=direct-index-resident pass=7 fail=33 not_applicable=0 worst_cell=occ8-direct-index-resident-absent-miss-wrapping-add worst_ratio=4.9794
strata_family dimension=distribution value=offset-beyond-direct pass=0 fail=40 not_applicable=0 worst_cell=occ16-offset-beyond-direct-absent-miss-replacement worst_ratio=9.0402
strata_family dimension=distribution value=seeded-random pass=0 fail=40 not_applicable=0 worst_cell=occ16-seeded-random-present-hit-wrapping-add worst_ratio=6.9319
strata_family dimension=distribution value=clustered-skewed pass=13 fail=27 not_applicable=0 worst_cell=occ1-clustered-skewed-present-hit-wrapping-add worst_ratio=8.3332
strata_family dimension=operation value=present-hit pass=0 fail=40 not_applicable=0 worst_cell=occ2-offset-beyond-direct-present-hit-wrapping-add worst_ratio=9.0397
strata_family dimension=operation value=absent-miss pass=1 fail=39 not_applicable=0 worst_cell=occ16-offset-beyond-direct-absent-miss-replacement worst_ratio=9.0402
strata_family dimension=operation value=existing-update pass=6 fail=34 not_applicable=0 worst_cell=occ16-seeded-random-existing-update-replacement worst_ratio=4.5435
strata_family dimension=operation value=admitted-fresh-insert pass=11 fail=21 not_applicable=0 worst_cell=occ4-offset-beyond-direct-admitted-fresh-insert-wrapping-add worst_ratio=1.8992
strata_family dimension=operation value=growth-insert pass=2 fail=6 not_applicable=0 worst_cell=occ16-offset-beyond-direct-growth-insert-wrapping-add worst_ratio=2.2020
strata_family dimension=occupancy value=1 pass=2 fail=30 not_applicable=0 worst_cell=occ1-clustered-skewed-present-hit-wrapping-add worst_ratio=8.3332
strata_family dimension=occupancy value=2 pass=5 fail=27 not_applicable=0 worst_cell=occ2-offset-beyond-direct-present-hit-wrapping-add worst_ratio=9.0397
strata_family dimension=occupancy value=4 pass=5 fail=27 not_applicable=0 worst_cell=occ4-offset-beyond-direct-present-hit-wrapping-add worst_ratio=8.1348
strata_family dimension=occupancy value=8 pass=4 fail=28 not_applicable=0 worst_cell=occ8-offset-beyond-direct-present-hit-replacement worst_ratio=8.1264
strata_family dimension=occupancy value=16 pass=4 fail=28 not_applicable=0 worst_cell=occ16-offset-beyond-direct-absent-miss-replacement worst_ratio=9.0402
strata_family dimension=merge value=replacement pass=10 fail=70 not_applicable=0 worst_cell=occ16-offset-beyond-direct-absent-miss-replacement worst_ratio=9.0402
strata_family dimension=merge value=wrapping-add pass=10 fail=70 not_applicable=0 worst_cell=occ2-offset-beyond-direct-present-hit-wrapping-add worst_ratio=9.0397
performance_registration registered=160 run_cells=160 source=matrix.json.current_strata identity=enforced accept=false
axis=identity status=pass evidence="identity row is unchanged by this measurement: the 17-key constant-coordinate witness admits all 17/17 distinct keys with oracle agreement and hashtrinity coalesces 17 submitted onto 1 live; the full-scale 1e8 coordinate-colliding run is unrun"
axis=round_trip status=boundary evidence="apeiron::Table exposes no encode/decode surface; the sentinel/boundary round-trip is not exercisable"
axis=determinism status=missing evidence="this measurement does not rerun the separate-process determinism check; see --check correctness"
axis=growth_merge_refusal status=pass evidence="region growth and the N=0 SizeOverflow refusal are observed in the correctness smoke; remove remains absent, so a performance measurement does not repair the remove row"
axis=performance status=boundary evidence="shared fixed-16 workload: initial_keys=4096 (16 per shard x 256 shards), lookups=1048576, updates=1048576, warmup=5, samples=31; replacement ratio=2.3785 candidate_ht_ratio=2.5477 candidate_current_ratio=1.0712 wrapping-add ratio=2.3639 candidate_ht_ratio=2.5608 candidate_current_ratio=1.0833; ratio_limit=1.00; profile=full; candidate/hashtrinity=2.5608 candidate/current=1.0833 (experimental, not the REG-1 axis); REG1-PERF-SHARED-FIXED16=2.378; strata: cells=160 measured=160 pass=20 fail=140 not_applicable=0 missing=0 rejected=0 resource_cells=3 campaigns=3 samples=15 ops=4096 ht_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc=\"rustc 1.97.0-nightly (17584a181 2026-04-13)\" worst_cell=occ16-offset-beyond-direct-absent-miss-replacement worst_ratio=9.0402"
axis=coverage status=missing evidence="coverage census and fractions are unrun"
report_complete=false
reg1_pass=false
control=placement-only-address observed=red reason="a placement-only TableAddress is not the production address for a coordinate word"
control=retired-shardfull-receipt observed=red cell=occ16-direct-index-resident-refusal-replacement reason="registered current cell occ16-direct-index-resident-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-direct-index-resident-refusal-replacement reason="registered current cell occ16-direct-index-resident-growth-insert-replacement oracle digest 500f3126b0d2c725 != current run 5035b5a9c61b9725"
control=retired-shardfull-receipt observed=red cell=occ16-direct-index-resident-refusal-wrapping-add reason="registered current cell occ16-direct-index-resident-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-direct-index-resident-refusal-wrapping-add reason="registered current cell occ16-direct-index-resident-growth-insert-wrapping-add oracle digest 500f3126b0d2c725 != current run 5035b5a9c61b9725"
control=retired-shardfull-receipt observed=red cell=occ16-offset-beyond-direct-refusal-replacement reason="registered current cell occ16-offset-beyond-direct-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-offset-beyond-direct-refusal-replacement reason="registered current cell occ16-offset-beyond-direct-growth-insert-replacement oracle digest 485754aa62d2c725 != current run eed487a5401b9725"
control=retired-shardfull-receipt observed=red cell=occ16-offset-beyond-direct-refusal-wrapping-add reason="registered current cell occ16-offset-beyond-direct-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-offset-beyond-direct-refusal-wrapping-add reason="registered current cell occ16-offset-beyond-direct-growth-insert-wrapping-add oracle digest 485754aa62d2c725 != current run eed487a5401b9725"
control=retired-shardfull-receipt observed=red cell=occ16-seeded-random-refusal-replacement reason="registered current cell occ16-seeded-random-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-seeded-random-refusal-replacement reason="registered current cell occ16-seeded-random-growth-insert-replacement oracle digest 5dc7a2ae90a03b25 != current run 1899f740eae78b25"
control=retired-shardfull-receipt observed=red cell=occ16-seeded-random-refusal-wrapping-add reason="registered current cell occ16-seeded-random-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-seeded-random-refusal-wrapping-add reason="registered current cell occ16-seeded-random-growth-insert-wrapping-add oracle digest 5dc7a2ae90a03b25 != current run 1899f740eae78b25"
control=retired-shardfull-receipt observed=red cell=occ16-clustered-skewed-refusal-replacement reason="registered current cell occ16-clustered-skewed-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-clustered-skewed-refusal-replacement reason="registered current cell occ16-clustered-skewed-growth-insert-replacement oracle digest 1cb22462026a4f25 != current run 226571a3ea055f25"
control=retired-shardfull-receipt observed=red cell=occ16-clustered-skewed-refusal-wrapping-add reason="registered current cell occ16-clustered-skewed-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-clustered-skewed-refusal-wrapping-add reason="registered current cell occ16-clustered-skewed-growth-insert-wrapping-add oracle digest 1cb22462026a4f25 != current run 226571a3ea055f25"
control=fabricated-current-cell observed=red reason="registered current cell occ1-direct-index-resident-present-hit-replacement-fabricated is absent from the current run"
control=placement-only-address observed=red reason="a placement-only TableAddress is not the production address for a coordinate word"
control=retired-shardfull-receipt observed=red cell=occ16-direct-index-resident-refusal-replacement reason="registered current cell occ16-direct-index-resident-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-direct-index-resident-refusal-replacement reason="registered current cell occ16-direct-index-resident-growth-insert-replacement oracle digest 500f3126b0d2c725 != current run 5035b5a9c61b9725"
control=retired-shardfull-receipt observed=red cell=occ16-direct-index-resident-refusal-wrapping-add reason="registered current cell occ16-direct-index-resident-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-direct-index-resident-refusal-wrapping-add reason="registered current cell occ16-direct-index-resident-growth-insert-wrapping-add oracle digest 500f3126b0d2c725 != current run 5035b5a9c61b9725"
control=retired-shardfull-receipt observed=red cell=occ16-offset-beyond-direct-refusal-replacement reason="registered current cell occ16-offset-beyond-direct-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-offset-beyond-direct-refusal-replacement reason="registered current cell occ16-offset-beyond-direct-growth-insert-replacement oracle digest 485754aa62d2c725 != current run eed487a5401b9725"
control=retired-shardfull-receipt observed=red cell=occ16-offset-beyond-direct-refusal-wrapping-add reason="registered current cell occ16-offset-beyond-direct-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-offset-beyond-direct-refusal-wrapping-add reason="registered current cell occ16-offset-beyond-direct-growth-insert-wrapping-add oracle digest 485754aa62d2c725 != current run eed487a5401b9725"
control=retired-shardfull-receipt observed=red cell=occ16-seeded-random-refusal-replacement reason="registered current cell occ16-seeded-random-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-seeded-random-refusal-replacement reason="registered current cell occ16-seeded-random-growth-insert-replacement oracle digest 5dc7a2ae90a03b25 != current run 1899f740eae78b25"
control=retired-shardfull-receipt observed=red cell=occ16-seeded-random-refusal-wrapping-add reason="registered current cell occ16-seeded-random-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-seeded-random-refusal-wrapping-add reason="registered current cell occ16-seeded-random-growth-insert-wrapping-add oracle digest 5dc7a2ae90a03b25 != current run 1899f740eae78b25"
control=retired-shardfull-receipt observed=red cell=occ16-clustered-skewed-refusal-replacement reason="registered current cell occ16-clustered-skewed-growth-insert-replacement operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-clustered-skewed-refusal-replacement reason="registered current cell occ16-clustered-skewed-growth-insert-replacement oracle digest 1cb22462026a4f25 != current run 226571a3ea055f25"
control=retired-shardfull-receipt observed=red cell=occ16-clustered-skewed-refusal-wrapping-add reason="registered current cell occ16-clustered-skewed-growth-insert-wrapping-add operation refusal != current run growth-insert"
control=retired-shardfull-digest observed=red cell=occ16-clustered-skewed-refusal-wrapping-add reason="registered current cell occ16-clustered-skewed-growth-insert-wrapping-add oracle digest 1cb22462026a4f25 != current run 226571a3ea055f25"
control=fabricated-current-cell observed=red reason="registered current cell occ1-direct-index-resident-present-hit-replacement-fabricated is absent from the current run"
spatial_protocol workload="71942db scene(n) packed u64, build+probe" sizes=[100000, 1000000] campaigns=3 warmup=2 samples=15 metric=elapsed_ns_median cpu="Intel(R) Core(TM) i9-14900KF" oracle=std::HashMap count folded in query order
spatial cell=spatial/n100000 population=100000 ops=99943 oracle_digest=5d5c572dd304446b selected_digest_ok=true selected_median_ns=19707218 selected_interval_ns=[18910431,31725503]
spatial cell=spatial/n100000 incumbent=hashtrinity_count_par selected_median_ns=19707218 incumbent_median_ns=2210482 incumbent_interval_ns=[1636037,6871075] ratio=8.9153 ratio_interval=[8.7423,9.0556] digest_ok=true workers=24 serial_fallback=false policy="hashtrinity par::Pool::global().width()=24; count_par serial fallback below 2^15=false"
spatial cell=spatial/n100000 incumbent=hashtrinity_get_each selected_median_ns=19707218 incumbent_median_ns=2184165 incumbent_interval_ns=[1613955,4391756] ratio=9.0228 ratio_interval=[8.9823,9.4479] digest_ok=true workers=1 serial_fallback=false policy="hashtrinity get_each single-core pipelined probe (P=16)"
spatial cell=spatial/n100000 incumbent=hashbrown_rayon selected_median_ns=19707218 incumbent_median_ns=2640717 incumbent_interval_ns=[2088205,5645096] ratio=7.4628 ratio_interval=[7.2338,7.6639] digest_ok=true workers=32 serial_fallback=false policy="rayon global pool, current_num_threads()=32"
spatial cell=spatial/n100000 incumbent=foldhash_par16 selected_median_ns=19707218 incumbent_median_ns=1166252 incumbent_interval_ns=[875013,2182814] ratio=16.8979 ratio_interval=[16.1193,17.8666] digest_ok=true workers=16 serial_fallback=false policy="16 std::thread::scope workers, foldhash sharded"
spatial cell=spatial/n100000 incumbent=fxhash_par16 selected_median_ns=19707218 incumbent_median_ns=980370 incumbent_interval_ns=[792395,1683929] ratio=20.1018 ratio_interval=[19.4473,21.1001] digest_ok=true workers=16 serial_fallback=false policy="16 std::thread::scope workers, fxhash sharded"
spatial cell=spatial/n1000000 population=1000000 ops=993259 oracle_digest=5e885edecf73faa3 selected_digest_ok=true selected_median_ns=1235053749 selected_interval_ns=[1212921016,1257102751]
spatial cell=spatial/n1000000 incumbent=hashtrinity_count_par selected_median_ns=1235053749 incumbent_median_ns=15892921 incumbent_interval_ns=[14607911,19912304] ratio=77.7109 ratio_interval=[77.1799,78.0214] digest_ok=true workers=24 serial_fallback=false policy="hashtrinity par::Pool::global().width()=24; count_par serial fallback below 2^15=false"
spatial cell=spatial/n1000000 incumbent=hashtrinity_get_each selected_median_ns=1235053749 incumbent_median_ns=15717175 incumbent_interval_ns=[13628420,19575948] ratio=78.5799 ratio_interval=[76.9795,81.6256] digest_ok=true workers=1 serial_fallback=false policy="hashtrinity get_each single-core pipelined probe (P=16)"
spatial cell=spatial/n1000000 incumbent=hashbrown_rayon selected_median_ns=1235053749 incumbent_median_ns=41241983 incumbent_interval_ns=[36540031,57077895] ratio=29.9465 ratio_interval=[28.4135,31.2551] digest_ok=true workers=32 serial_fallback=false policy="rayon global pool, current_num_threads()=32"
spatial cell=spatial/n1000000 incumbent=foldhash_par16 selected_median_ns=1235053749 incumbent_median_ns=8521887 incumbent_interval_ns=[7182705,17089069] ratio=144.9273 ratio_interval=[129.2260,147.2329] digest_ok=true workers=16 serial_fallback=false policy="16 std::thread::scope workers, foldhash sharded"
spatial cell=spatial/n1000000 incumbent=fxhash_par16 selected_median_ns=1235053749 incumbent_median_ns=9367241 incumbent_interval_ns=[7309491,21268279] ratio=131.8482 ratio_interval=[130.3004,133.1571] digest_ok=true workers=16 serial_fallback=false policy="16 std::thread::scope workers, fxhash sharded"
acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2
acceptance_registration incumbent=hashtrinity_table class=load-bearing adapter=built applicability=applicable symbol="hashtrinity::Table<StrataKey,i64>" adapter_reason="adapter built in this harness" applicability_reason="current-strata serial map incumbent on the same stored-key face"
acceptance_registration incumbent=hashtrinity_count_par class=load-bearing adapter=built applicability=authorized-not-applicable symbol="hashtrinity::FreqMap::count_par + get_each (packed u64)" adapter_reason="adapter built in this harness" applicability_reason="per-cell: the packed spatial count workload only; the fixed-shard strata cells are the serial-map face"
acceptance_registration incumbent=hashtrinity_get_each class=load-bearing adapter=built applicability=authorized-not-applicable symbol="hashtrinity::FreqMap::get_each (packed u64 probe)" adapter_reason="adapter built in this harness" applicability_reason="per-cell: the packed spatial count workload only; the fixed-shard strata cells are the serial-map face"
acceptance_registration incumbent=hashbrown_foldhash class=load-bearing adapter=built applicability=applicable symbol="hashbrown::HashMap<StrataKey,i64,foldhash>" adapter_reason="adapter built in this harness" applicability_reason="opaque-key exact map face; the same hit/miss/update/insert surface"
acceptance_registration incumbent=hashbrown_fxhash class=load-bearing adapter=built applicability=applicable symbol="hashbrown::HashMap<StrataKey,i64,fxhash>" adapter_reason="adapter built in this harness" applicability_reason="opaque-key exact map face; the same hit/miss/update/insert surface"
acceptance_registration incumbent=hashbrown_rayon class=load-bearing adapter=built applicability=authorized-not-applicable symbol="hashbrown::HashMap + rayon par_iter" adapter_reason="adapter built in this harness" applicability_reason="per-cell: the packed spatial count workload only; the fixed-shard strata cells are the serial-map face"
acceptance_registration incumbent=std_siphash class=load-bearing adapter=built applicability=applicable symbol="std::collections::HashMap<StrataKey,i64> (SipHash-1-3)" adapter_reason="adapter built in this harness" applicability_reason="correctness/determinism reference on the same total-map face"
acceptance_registration incumbent=foldhash_par16 class=load-bearing adapter=built applicability=authorized-not-applicable symbol="foldhash par16" adapter_reason="adapter built in this harness" applicability_reason="per-cell: the packed spatial count workload only; the fixed-shard strata cells are the serial-map face"
acceptance_registration incumbent=fxhash_par16 class=load-bearing adapter=built applicability=authorized-not-applicable symbol="fxhash par16" adapter_reason="adapter built in this harness" applicability_reason="per-cell: the packed spatial count workload only; the fixed-shard strata cells are the serial-map face"
acceptance_registration incumbent=xxhash_xxh3 class=informational adapter=built applicability=applicable symbol="twox-hash XXH3" adapter_reason="adapter built in this harness" applicability_reason="informational hasher-quality comparator"
acceptance_registration incumbent=xxhash_xxh64 class=informational adapter=built applicability=applicable symbol="twox-hash XXH64" adapter_reason="adapter built in this harness" applicability_reason="informational hasher-quality comparator"
acceptance_registration incumbent=wyhash class=informational adapter=built applicability=applicable symbol="wyhash" adapter_reason="adapter built in this harness" applicability_reason="informational hasher-quality comparator"
acceptance_registration incumbent=ahash class=informational adapter=built applicability=applicable symbol="ahash" adapter_reason="adapter built in this harness" applicability_reason="informational hasher-quality comparator"
acceptance_registration incumbent=fnv class=informational adapter=built applicability=applicable symbol="fnv-1a" adapter_reason="adapter built in this harness" applicability_reason="informational hasher-quality comparator"
acceptance_registration incumbent=blake3 class=informational adapter=built applicability=applicable symbol="blake3" adapter_reason="adapter built in this harness" applicability_reason="informational cryptographic comparator"
acceptance_registration incumbent=sha256 class=informational adapter=built applicability=applicable symbol="sha2 SHA-256" adapter_reason="adapter built in this harness" applicability_reason="informational cryptographic comparator"
acceptance_ratio incumbent=hashtrinity_table class=load-bearing pairs=160 digest_fail=0 r_min=0.2884 r_max=9.2695
acceptance_ratio incumbent=hashtrinity_count_par class=load-bearing pairs=2 digest_fail=0 r_min=8.7423 r_max=78.0214
acceptance_ratio incumbent=hashtrinity_get_each class=load-bearing pairs=2 digest_fail=0 r_min=8.9823 r_max=81.6256
acceptance_ratio incumbent=hashbrown_foldhash class=load-bearing pairs=160 digest_fail=0 r_min=0.3779 r_max=16.1926
acceptance_ratio incumbent=hashbrown_fxhash class=load-bearing pairs=160 digest_fail=0 r_min=0.4578 r_max=29.5121
acceptance_ratio incumbent=hashbrown_rayon class=load-bearing pairs=2 digest_fail=0 r_min=7.2338 r_max=31.2551
acceptance_ratio incumbent=std_siphash class=load-bearing pairs=160 digest_fail=0 r_min=0.1600 r_max=3.8239
acceptance_ratio incumbent=foldhash_par16 class=load-bearing pairs=2 digest_fail=0 r_min=16.1193 r_max=147.2329
acceptance_ratio incumbent=fxhash_par16 class=load-bearing pairs=2 digest_fail=0 r_min=19.4473 r_max=133.1571
acceptance_ratio incumbent=xxhash_xxh3 class=informational pairs=160 digest_fail=0 r_min=0.0488 r_max=2.3875
acceptance_ratio incumbent=xxhash_xxh64 class=informational pairs=160 digest_fail=0 r_min=0.3339 r_max=9.5952
acceptance_ratio incumbent=wyhash class=informational pairs=160 digest_fail=0 r_min=0.3895 r_max=14.6033
acceptance_ratio incumbent=ahash class=informational pairs=160 digest_fail=0 r_min=0.3528 r_max=13.1617
acceptance_ratio incumbent=fnv class=informational pairs=160 digest_fail=0 r_min=0.3232 r_max=8.1112
acceptance_ratio incumbent=blake3 class=informational pairs=160 digest_fail=0 r_min=0.0231 r_max=0.4897
acceptance_ratio incumbent=sha256 class=informational pairs=160 digest_fail=0 r_min=0.0286 r_max=0.7166
acceptance_rule selected/incumbent <= 1.00 on every applicable (cell, incumbent) pair; the campaign interval max must be <= 1.00
acceptance_incumbent id=hashtrinity_table class=load-bearing adapter=built applicable=162 pass=14 fail=146 missing=0 not_applicable=2 informational=0
acceptance_incumbent id=hashtrinity_count_par class=load-bearing adapter=built applicable=162 pass=0 fail=2 missing=0 not_applicable=160 informational=0
acceptance_incumbent id=hashtrinity_get_each class=load-bearing adapter=built applicable=162 pass=0 fail=2 missing=0 not_applicable=160 informational=0
acceptance_incumbent id=hashbrown_foldhash class=load-bearing adapter=built applicable=162 pass=11 fail=149 missing=0 not_applicable=2 informational=0
acceptance_incumbent id=hashbrown_fxhash class=load-bearing adapter=built applicable=162 pass=10 fail=150 missing=0 not_applicable=2 informational=0
acceptance_incumbent id=hashbrown_rayon class=load-bearing adapter=built applicable=162 pass=0 fail=2 missing=0 not_applicable=160 informational=0
acceptance_incumbent id=std_siphash class=load-bearing adapter=built applicable=162 pass=98 fail=62 missing=0 not_applicable=2 informational=0
acceptance_incumbent id=foldhash_par16 class=load-bearing adapter=built applicable=162 pass=0 fail=2 missing=0 not_applicable=160 informational=0
acceptance_incumbent id=fxhash_par16 class=load-bearing adapter=built applicable=162 pass=0 fail=2 missing=0 not_applicable=160 informational=0
acceptance_incumbent id=xxhash_xxh3 class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_incumbent id=xxhash_xxh64 class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_incumbent id=wyhash class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_incumbent id=ahash class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_incumbent id=fnv class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_incumbent id=blake3 class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_incumbent id=sha256 class=informational adapter=built applicable=0 pass=0 fail=0 missing=0 not_applicable=0 informational=162
acceptance_summary registered=2592 load_bearing=1458 applicable=650 pass=133 fail=517 missing=0 unauthorized=0 not_applicable=808 informational=1134 worst_pair=spatial/n1000000/foldhash_par16 worst_ratio=147.2329 full_acceptance=false
acceptance_first_failure=(occ1-direct-index-resident-present-hit-replacement, hashtrinity_table) rmax=4.0406 > 1.00
control=wrong-workload-identity observed=red child_exit=2
control=invalid-samples observed=red child_exit=2
control=absent-samples observed=red child_exit=2
control=cached-home-only-collision observed=red child_exit=2
control=wrong-oracle-digest observed=red child_exit=2
control=deleted-required-current-cell observed=red child_exit=2
control=selected-implementation-mismatch observed=red child_exit=2
control=strict-acceptance-single-failing-cell observed=red child_exit=2
control=spatial-wrong-packed-input observed=red child_exit=2
control=spatial-wrong-bucket-routing observed=red child_exit=2
control=spatial-wrong-worker-width observed=red child_exit=2
control=acceptance-failing-non-hashtrinity-pair observed=red child_exit=2
control=acceptance-omitted-incumbent-pair observed=red child_exit=2
control=acceptance-unauthorized-not-applicable observed=red child_exit=2
control=acceptance-crossing-ratio-interval observed=red child_exit=2
control=acceptance-wrong-oracle-digest observed=red child_exit=2
control=acceptance-serial-adapter-wrong-result observed=red child_exit=2

## Global baseline receipt (APEIRON-PERF-GLOBAL-W7-T1)

Binding: this receipt binds to the dirty-tree digests below, **not to HEAD**.
`apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`. The captured
`src/table.rs` and `src/table/scan.rs` are uncommitted; HEAD's copies differ. The
flat-resident rework is compared against the digests in §1, not against a commit.

### 1 · Source and harness snapshot

| artifact | sha256 | lines |
|---|---|---|
| `src/table.rs` | `e56014dade8f385a83f1a2c0721fb186a5d1bd580a64486c5ef36ad17d2d17ea` | 1330 |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` | 210 |
| `tests/table.rs` | `12ad029f27c0e982c44fe4248797b3b0dc6bb2248c0bebd7d18527221844bab5` | 356 |
| `examples/reg1/main.rs` | `51e1c10c2fbff6af48cddf82bd63ae3f3f2df318bb20b6c189517a6ebd0ac46c` | 620 |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` | 1465 |
| `examples/reg1/performance.rs` | `ab92dfc2ec9d2ce4944acece2e4b96fb4c0d2dfe714d3e5f97f64e79db7c7bae` | 1193 |
| `examples/reg1/strata.rs` | `fdf56add117ccb1da0c413bc0be43063b9ed3f8975488de6f3b641f1993da90b` | 1498 |
| `examples/reg1/acceptance.rs` | `183c019c8c9447e1518b2c76ce577eb910b0e92bb24fa302a74576e34e88f3c0` | 1102 |
| `examples/reg1/history.rs` | `46353c85ad4352c59f7ec9c3e5f2a42df532adfd3a2fbafc0379025cc11af057` | 1498 |
| `examples/reg1/adapters/serial.rs` | `b7fa9a735ad673b46899e0eabb049db63b720cede09de2e8ea98b11d33c744b3` | 356 |
| `examples/reg1/adapters/spatial.rs` | `e22cc9a471e5075241cf2eb509cf298ba8e8c1f3dc13270bbd368c212bac3ba2` | 851 |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` | 1636 |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` | 151 |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` | 899 |

The run does not write any of these files; re-checked after the run, digests
unchanged.

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux (rolling), kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target | `x86_64-linux`, release |
| serial harness threads | `1` (`single_thread=true`, from the run's own header) |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| apeiron head | `c450c222905c050f793a23332708c230c658f82c` (dirty) |

### 3 · Run and result

Command (cwd `/home/socol/Workspace/apeiron`):

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Exit code: **1** (strict acceptance failure; the run is the evidence, nothing was
fixed). A cargo warning `error: duplicate key` from the hashtrinity git checkout
(`opsfold/Cargo.toml:61`) is printed before `Finished`; it is non-fatal and does
not determine the exit status.

Failure class (stderr, last line):

```text
acceptance failed: applicable=650 failing=517 missing=0 unauthorized=0 full_acceptance=false
```

`acceptance_summary`:

```text
acceptance_summary registered=2592 load_bearing=1458 applicable=650 pass=133 fail=517 missing=0 unauthorized=0 not_applicable=808 informational=1134 worst_pair=spatial/n1000000/foldhash_par16 worst_ratio=147.2329 full_acceptance=false
```

`acceptance_first_failure`:

```text
acceptance_first_failure=(occ1-direct-index-resident-present-hit-replacement, hashtrinity_table) rmax=4.0406 > 1.00
```

`strata_summary` (the 160-cell serial fixed-shard axis):

```text
strata_summary cells=160 measured=160 pass=20 fail=140 not_applicable=0 missing=0 rejected=0 resource_cells=3 resource_not_applicable=1 resource_missing=2 campaigns=3 warmup=2 samples=15 ops=4096 ht_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc="rustc 1.97.0-nightly (17584a181 2026-04-13)" cpu="Intel(R) Core(TM) i9-14900KF"
```

`report_complete=false`, `reg1_pass=false`.

### 4 · Per-incumbent ratios and pass/fail (all registered incumbents)

`pairs` is the count of measured pairs whose oracle digest agreed; `applicable` is
the tally's evaluated-pair count, with `pass`/`fail`/`not_applicable` summing to it.
Load-bearing rows carry the acceptance verdict.

Verbatim per-incumbent ratio lines:

```text
acceptance_ratio incumbent=hashtrinity_table class=load-bearing pairs=160 digest_fail=0 r_min=0.2884 r_max=9.2695
acceptance_ratio incumbent=hashtrinity_count_par class=load-bearing pairs=2 digest_fail=0 r_min=8.7423 r_max=78.0214
acceptance_ratio incumbent=hashtrinity_get_each class=load-bearing pairs=2 digest_fail=0 r_min=8.9823 r_max=81.6256
acceptance_ratio incumbent=hashbrown_foldhash class=load-bearing pairs=160 digest_fail=0 r_min=0.3779 r_max=16.1926
acceptance_ratio incumbent=hashbrown_fxhash class=load-bearing pairs=160 digest_fail=0 r_min=0.4578 r_max=29.5121
acceptance_ratio incumbent=hashbrown_rayon class=load-bearing pairs=2 digest_fail=0 r_min=7.2338 r_max=31.2551
acceptance_ratio incumbent=std_siphash class=load-bearing pairs=160 digest_fail=0 r_min=0.1600 r_max=3.8239
acceptance_ratio incumbent=foldhash_par16 class=load-bearing pairs=2 digest_fail=0 r_min=16.1193 r_max=147.2329
acceptance_ratio incumbent=fxhash_par16 class=load-bearing pairs=2 digest_fail=0 r_min=19.4473 r_max=133.1571
```

| incumbent | class | pairs | r_min | r_max | applicable | pass | fail | not_applicable |
|---|---|---|---|---|---|---|---|---|
| `hashtrinity_table` | load-bearing | 160 | 0.2884 | 9.2695 | 162 | 14 | 146 | 2 |
| `hashtrinity_count_par` | load-bearing | 2 | 8.7423 | 78.0214 | 162 | 0 | 2 | 160 |
| `hashtrinity_get_each` | load-bearing | 2 | 8.9823 | 81.6256 | 162 | 0 | 2 | 160 |
| `hashbrown_foldhash` | load-bearing | 160 | 0.3779 | 16.1926 | 162 | 11 | 149 | 2 |
| `hashbrown_fxhash` | load-bearing | 160 | 0.4578 | 29.5121 | 162 | 10 | 150 | 2 |
| `hashbrown_rayon` | load-bearing | 2 | 7.2338 | 31.2551 | 162 | 0 | 2 | 160 |
| `std_siphash` | load-bearing | 160 | 0.1600 | 3.8239 | 162 | 98 | 62 | 2 |
| `foldhash_par16` | load-bearing | 2 | 16.1193 | 147.2329 | 162 | 0 | 2 | 160 |
| `fxhash_par16` | load-bearing | 2 | 19.4473 | 133.1571 | 162 | 0 | 2 | 160 |
| `xxhash_xxh3` | informational | 160 | 0.0488 | 2.3875 | 0 | 0 | 0 | 0 (informational=162) |
| `xxhash_xxh64` | informational | 160 | 0.3339 | 9.5952 | 0 | 0 | 0 | 0 (informational=162) |
| `wyhash` | informational | 160 | 0.3895 | 14.6033 | 0 | 0 | 0 | 0 (informational=162) |
| `ahash` | informational | 160 | 0.3528 | 13.1617 | 0 | 0 | 0 | 0 (informational=162) |
| `fnv` | informational | 160 | 0.3232 | 8.1112 | 0 | 0 | 0 | 0 (informational=162) |
| `blake3` | informational | 160 | 0.0231 | 0.4897 | 0 | 0 | 0 | 0 (informational=162) |
| `sha256` | informational | 160 | 0.0286 | 0.7166 | 0 | 0 | 0 | 0 (informational=162) |

All `digest_fail=0`. The 160-pair incumbents are the serial fixed-shard strata
face; the 2-pair incumbents are the packed spatial `n100000`/`n1000000` face. The
`not_applicable` counts are the registered per-face exclusions (serial incumbents on
the spatial face and spatial incumbents on the strata face), not missing adapters
(`missing=0`).

### 5 · Spatial selected-vs-incumbent medians

`spatial_protocol` sizes `[100000, 1000000]`, campaigns `3`, warmup `2`, samples
`15`, metric `elapsed_ns_median`, oracle `std::HashMap` count folded in query order.

#### `spatial/n100000` — selected `selected_median_ns=19707218`

| incumbent | selected ns | incumbent ns | ratio | ratio_interval |
|---|---|---|---|---|
| `hashtrinity_count_par` | 19707218 | 2210482 | 8.9153 | [8.7423, 9.0556] |
| `hashtrinity_get_each` | 19707218 | 2184165 | 9.0228 | [8.9823, 9.4479] |
| `hashbrown_rayon` | 19707218 | 2640717 | 7.4628 | [7.2338, 7.6639] |
| `foldhash_par16` | 19707218 | 1166252 | 16.8979 | [16.1193, 17.8666] |
| `fxhash_par16` | 19707218 | 980370 | 20.1018 | [19.4473, 21.1001] |

#### `spatial/n1000000` — selected `selected_median_ns=1235053749`

| incumbent | selected ns | incumbent ns | ratio | ratio_interval |
|---|---|---|---|---|
| `hashtrinity_count_par` | 1235053749 | 15892921 | 77.7109 | [77.1799, 78.0214] |
| `hashtrinity_get_each` | 1235053749 | 15717175 | 78.5799 | [76.9795, 81.6256] |
| `hashbrown_rayon` | 1235053749 | 41241983 | 29.9465 | [28.4135, 31.2551] |
| `foldhash_par16` | 1235053749 | 8521887 | 144.9273 | [129.2260, 147.2329] |
| `fxhash_par16` | 1235053749 | 9367241 | 131.8482 | [130.3004, 133.1571] |

`selected_digest_ok=true` and `digest_ok=true` on every spatial cell.

### 6 · Baseline deficits (this is the "before")

Serial (160 strata cells, 3 campaigns): the measured serial worst against
`hashbrown_fxhash` is `r_max=29.5121`; `std_siphash` is the closest incumbent at
`r_max=3.8239`. Spatial at `n=1000000`: `hashtrinity_count_par` median ratio
`77.7109` (interval max `78.0214`) and `foldhash_par16` median ratio `144.9273`
(interval max `147.2329`).

The task handoff states the same two deficits with prior approximate figures:
serial direct-index-resident worst ~28x behind `hashbrown_fxhash`; spatial n=1M
79.9x behind `hashtrinity_count_par` and 138x behind `foldhash_par16`;
`std_siphash` closest at 3.77. Those figures are prior measurements from a
different run moment; the live figures in this section are authoritative for this
receipt's digests. Measured deltas: hashbrown_fxhash +1.5, count_par −2.2,
foldhash_par16 +6.9, std_siphash +0.05. No speed claim is made.

The per-incumbent ratio print does not name the serial cell carrying the
`hashbrown_fxhash` maximum; the earlier no-spatial run attributed the serial
`hashbrown_fxhash` worst to `occ16-clustered-skewed-absent-miss-replacement`
(ratio `28.2913`). This receipt records the aggregate `r_max=29.5121` only.

### 7 · Controls

Observed: **53 red control lines, 0 green** (22 distinct names). Distinct counts:
`retired-shardfull-digest` ×16, `retired-shardfull-receipt` ×16,
`fabricated-current-cell` ×2, `placement-only-address` ×2, and one each of
`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`,
`deleted-required-current-cell`, `selected-implementation-mismatch`,
`strict-acceptance-single-failing-cell`, `spatial-wrong-packed-input`,
`spatial-wrong-bucket-routing`, `spatial-wrong-worker-width`,
`acceptance-failing-non-hashtrinity-pair`, `acceptance-omitted-incumbent-pair`,
`acceptance-unauthorized-not-applicable`, `acceptance-crossing-ratio-interval`,
`acceptance-wrong-oracle-digest`, `acceptance-serial-adapter-wrong-result`. Every
acceptance/performance control child exits 2.

### 8 · Raw run logs

| artifact | sha256 | lines |
|---|---|---|
| `docs/evidence/perf-global-baseline.log` (stdout) | `92badfedd7f756d903119bce389a926fab1fa7fea27f4af4f843705c2277d62d` | 492 |
| `docs/evidence/perf-global-baseline.stderr.log` (stderr) | `9f7a6ec3d036a2947145c25408b67ab133740e4b9c741bc17102bd6b8e91e43d` | 8 |

### 9 · Scope and exclusions

- In scope: identity of the pre-rework `src/table.rs`/`src/table/scan.rs` and
  harness, the strict per-pair acceptance result, and the serial + spatial ratios
  on this machine/toolchain. This is the "before" for the flat-resident rework.
- Out of scope (obligations not evidenced here): `Sigma` in every executing
  signature, `remove`, SIMD, allocator behaviour, threading memory semantics,
  std internals, rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V. The spatial
  `std::thread::scope` orchestration is an unextracted boundary.
- No history, Kani, `proof/regen.sh`, or `lake build` was run for this receipt.
- Grade: snapshot, harness and log identity are digest receipts (`tested`
  identity); the run is a `measured`/gate result on the recorded machine and tool
  identity.

### 10 · Open items

- **REG-1 OPEN**: the strict performance acceptance exits 1 with
  `full_acceptance=false`, 517 of 650 applicable pairs failing.
- **G1 OPEN**: the performance objective is not met by the current representation;
  the serial and spatial deficits in §6 are the baseline to be reduced.

error: duplicate key
  --> ../../.cargo/git/checkouts/fold-62126b3be665e44a/4cc1447/opsfold/Cargo.toml:61:1
   |
61 | netfold = { path = "../netfold", optional = true }
   | ^^^^^^^
    Finished `release` profile [optimized] target(s) in 0.13s
     Running `target/release/examples/reg1 --check performance --profile full --accept --controls`
acceptance failed: applicable=650 failing=517 missing=0 unauthorized=0 full_acceptance=false

## Micro spatial phase split, post stable scatter (APEIRON-PERF-SCATTER-W3-T1)

Focused release micro-benchmark of the packed spatial build+probe face on the current
source, after the stable chunk scatter landed in `src/table/partition.rs` and the
reserve-all integration in `src/table/parallel.rs`. No source, test, proof or example
file was edited; this run measures the committed harness against the working source.

### 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.
The run writes none of these files.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `5f2608010137b16b195baad62cb5214bcadeeea13a9df4da39e19f4bf543a61e` |
| `src/table/partition.rs` | `7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64` |
| `src/table/parallel.rs` | `b22650c1c6db76a7865e782fc0fc56174d3b8b6ac2c7eaccc0af087815df193f` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `examples/micro_spatial.rs` | `40b374540eb1c6ca1a6d13396eb916d2873a13bef9985baa38676b973b18d2df` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since [`perf-ledger.md`](perf-ledger.md):
`src/table.rs` (`79621926…` → `5f260801…`), `src/table/partition.rs`
(`2e9a2f56…` → `7cf47244…`), `src/table/parallel.rs` (new to the binding table).

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads, ~4.0 GHz nominal |
| governor | `powersave` |
| memory | 62 GiB total, 31 GiB available at run |
| load average | 0.99 / 1.01 / 0.90 at run start; no Lean, Kani or other build concurrent |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| run log (stdout) | `/tmp/opencode/micro-spatial-20260919-204725.log`, sha256 `0c51eecfef1620cad19535a3c15c2b8a34f4afb685c4308501c8cb874c488f4f`, 7 lines |
| run log (stderr) | `/tmp/opencode/micro-spatial-20260919-204725.log.stderr`, sha256 `5450e94afb4cdc734698eb64504c8a512eec276e0eba039d3bb416cc466a2d4d` |

Non-fatal cargo warning `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61` precedes `Finished` and does not set the exit status.

### 3 · Command, exit, elapsed

```sh
# cwd /home/socol/Workspace/apeiron
/usr/bin/time -v nice -n 10 timeout 1800 \
  cargo run --release --locked --example micro_spatial -- --controls
# exit 0; wall 0:01.34; user 6.08 s; sys 0.48 s; max RSS 152716 KiB; timeout 1800 NOT reached
```

The harness protocol: `71942db scene(n)` packed-u64 build+probe, `workers=16`,
`warmup=2`, `samples=15`, `metric=elapsed_ns_median`, oracle `std::HashMap` count
folded in query order. `digest_ok=true` for both sizes: the FNV digest of the probe
readback equals the independent oracle.

The `--controls` argument is inert in this harness: `examples/micro_spatial.rs` reads
no `std::env::args`, so the flag changes nothing. The only in-harness check is the
printed `digest_ok`.

### 4 · Phase table (median ns, workers=16)

Each phase value is the median of the 15 timed samples after 2 warmups. Percentages
are the harness's own display: each field over the six-field subtotal it tracks. The
`reserve` phase (`PhaseNanos::reserve`, added with reserve-all) is not read by this
harness, so it is absent from both the printed total and the percentages.

#### Build

| cell | qualify | partition | validate | schedule | kernel | assemble | six-field total | build median |
|---|---|---|---|---|---|---|---|---|
| `spatial/n100000` | 95596 (6.5%) | 470045 (32.1%) | 316854 (21.7%) | 62135 (4.2%) | 511732 (35.0%) | 6166 (0.4%) | 1462528 | 1675838 |
| `spatial/n1000000` | 1481721 (6.1%) | 3959142 (16.2%) | 6548069 (26.8%) | 63224 (0.3%) | 12352666 (50.6%) | 10294 (0.0%) | 24415116 | 27671723 |

#### Probe

| cell | qualify | partition | validate | schedule | kernel | assemble | six-field total | probe median |
|---|---|---|---|---|---|---|---|---|
| `spatial/n100000` | 118028 (7.4%) | 317425 (19.8%) | 151052 (9.4%) | 60780 (3.8%) | 460051 (28.7%) | 495233 (30.9%) | 1602569 | 1777546 |
| `spatial/n1000000` | 2344010 (9.4%) | 2411899 (9.7%) | 4478654 (18.0%) | 64292 (0.3%) | 11164580 (44.9%) | 4388490 (17.7%) | 24851925 | 24654940 |

#### End-to-end

| cell | population | ops | build median ns | probe median ns | end-to-end median ns | ns/op |
|---|---|---|---|---|---|---|
| `spatial/n100000` | 100000 | 99943 | 1675838 | 1777546 | 3478495 | 34.7850 |
| `spatial/n1000000` | 1000000 | 993259 | 27671723 | 24654940 | 52820140 | 52.8201 |

Unprinted `reserve` + accounting gap, from `build/probe median − six-field total`:
n100k build 213310 ns (~12.7%), n1M build 3256607 ns (~11.8%), n100k probe 174977 ns
(~9.8%), n1M probe −196985 ns (median non-additivity, within sample noise). The
`reserve` phase is material at ~12% of the build and is not in the harness output.

### 5 · Partition/validate before and after

The "before" row is the historical anchor stated in
[`docs/epics/parallel-stable-scatter.md`](../epics/parallel-stable-scatter.md)
(Problem): partition plus `Parts::is_intact` about 28% of the spatial build, per-shard
kernel 46–55%, assemble about 0% build / 16% probe. That anchor carries no absolute ns
and is **not** a fresh W0 measurement; no W0 receipt exists in `docs/evidence/`. The
"after" row is §4. Post-scatter `partition` now includes the checked chunk plan,
scatter-scratch reservation, stable scatter and stable concatenation, so part of the
kernel→partition shift is re-attribution, not pure added cost.

| run | kernel before | kernel after | partition+validate before | partition+validate after | assemble before | assemble after |
|---|---|---|---|---|---|---|
| build n100k | 46–55% | 35.0% | ~28% | 53.8% | ~0% | 0.4% |
| build n1M | 46–55% | 50.6% | ~28% | 43.0% | ~0% | 0.0% |
| probe n100k | 46–55% | 28.7% | ~28% | 29.2% | ~16% | 30.9% |
| probe n1M | 46–55% | 44.9% | ~28% | 27.7% | ~16% | 17.7% |

Direction: build `partition+validate` rose (28% → 43–54%) and build `kernel` fell
(46–55% → 35–51%) as the scatter cost moved into `partition`. Probe
`partition+validate` is roughly unchanged (27.7–29.2%); probe `assemble` rose to
17.7–30.9%. No speedup claim is made; the strict pair gate is the acceptance path.

### 6 · Measured vs not

- **Measured**: the phase medians, six-field totals, end-to-end build+probe medians and
  ns/op in §4, on this machine, toolchain, dirty-source identity and single process
  invocation, with the internal oracle digest matching (`digest_ok=true`). Grade:
  `measured`.
- **Interval**: none. The harness prints one median per phase from 15 samples; it does
  not print min/max, and this receipt is one invocation, not repeated campaigns. The
  numbers are single-run medians.
- **Not measured / out of scope**: strict REG-1 (`examples/reg1`, `--check performance
  --profile full --accept --controls`), `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, the `reserve` phase (not read by the
  harness), the W3 digest-corruption rejecting control, cross-run intervals, and the
  `std::thread::scope` executor boundary (spawn/join/unwind/liveness remain
  unextracted). This diagnostic cannot pass G1.
- **Grade**: `measured` for §4; `anchor` for the §5 before column; the digests and log
  hashes are identity receipts.

### 7 · Residual and next bounded task

- Residual: the strict spatial pair gate
  (`cargo run --release --locked --example reg1 -- --check performance --profile full
  --accept --controls`) is pending; every spatial pair must be recorded and no interval
  maximum above `1.0` closes G1. Also pending: a fresh pre-scatter baseline (the W0
  receipt is absent, so §5's before column is the historical anchor), and the W3
  digest-corruption rejecting control, which was not run.
- Next bounded task: APEIRON-PERF-SCATTER-W4-T1, run the strict full-profile registry
  and record every spatial pair (including concurrency/address-face outcomes and every
  named incumbent) with its intervals and the raw-log hash. Do not weaken the `<= 1.0`
  contract.

## REG-1 performance remeasurement, post correctness lifecycle (APEIRON-PERF-REMEASURE-T3)

Fresh strict-gate measurement on the current source, after the spill tag
fingerprint (`tag = TAG_LIVE | ((place >> 8) as u8 & 0x7f)`), the checked
fallible receipted summary lifecycle, the fallible density planner, the parallel
serial reserve-all + seam, and the intact-flag `is_intact`. The prior receipt
([`perf-ledger.md`](perf-ledger.md))
binds `src/table.rs` sha256 `79621926…`; the executing source is now
`00c3c7f8…`. The prior numbers are stale for this identity; this receipt
supersedes them for the current identity only.

### 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `00c3c7f82c840bd6be18978be5f356d97c3918278f373396f4a0c9417ae65e01` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/partition.rs` | `7207a4e9784b5c4153856f2c73d1d78df3dadf25a617d6aae809670cac4b2bf2` |
| `src/table/parallel.rs` | `66f0ba73450cd834da9f483403d03aa20503374cff5765e54c7ed8562d52ed46` |
| `src/table/resources.rs` | `a1d6e242c912706752375027c21b255677830ec74254d89fc8d5dd1324c06009` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the post-spill receipt: `src/table.rs` (`79621926…` → `00c3c7f8…`),
`src/table/partition.rs` (`2e9a2f56…` → `7207a4e9…`),
`src/table/resources.rs` (`5b04be96…` → `a1d6e242…`); `src/table/parallel.rs` is
new to the binding table; `src/table/scan.rs` is unchanged (`698b1eca…`). The run
writes none of these files. The reg1 harness sources (`main.rs`, `performance.rs`,
`acceptance.rs`, `strata.rs`, `matrix.rs`, `matrix.json`, `Cargo.toml`,
`Cargo.lock`) are byte-identical to the post-spill receipt.

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| CPU governor / turbo | `powersave`; `intel_pstate/no_turbo=0` |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run log | `/tmp/opencode/reg1-full-postcorrect-20260919-180612.log`, sha256 `c8d43f2cd165c251b20e02170820d79e2ba31bc2aeb27565455b4df2197d924b`, 2690 lines |

The host was otherwise idle: load average `0.65/0.86/1.20` at start and
`0.67/0.87/1.14` after the run; no Lean, Kani, coverage or other build ran
concurrently.

### 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.50; user 11.71 s; max RSS 518900 KiB
# log /tmp/opencode/reg1-build-postcorrect-20260919-180607.log
#     sha256 1f5bf417f456012a92e9b331127b08cc69a3a41c29ed57f4f23a4e80a08760cf

# 2. strict full gate (cwd /home/socol/Workspace/apeiron)
TS=20260919-180612
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-postcorrect-$TS.log 2>&1
# exit 1 (strict acceptance failure); wall 0:01:40.11; user 191.82 s; system 7.87 s;
# 199% CPU; max RSS 714868 KiB. timeout 3600 NOT reached; smoke fallback not run.
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

### 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=421 fail=1029 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 1023 are ratio failures. Against the post-spill
receipt (`pass=387 fail=1063`, same `registered`/`load_bearing`/`applicable`):
`pass` rose 387 → 421, `fail` fell 1063 → 1029.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).
The pooled point ratio can fall marginally outside the campaign min/max interval
(observed on several cells, both directions); acceptance reads the interval max.

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

### 5 · Worst failing pairs (top-20 current-strata, by interval max)

`ns/op` is `sel_median_ns / 4096`.

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ16-seeded-random-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 20.4682 | [20.0871,20.5922] | 16.161 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 19.8637 | [19.1943,20.2727] | 18.923 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 19.1104 | [18.5575,19.3316] | 18.812 |
| current-strata | `occ16-seeded-random-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 18.2012 | [18.1236,18.5058] | 15.993 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_foldhash w1` | 14.9341 | [14.8725,15.0975] | 18.923 |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 14.6136 | [14.5484,14.6337] | 10.949 |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 14.5840 | [14.5000,14.6471] | 10.931 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_foldhash w1` | 14.1460 | [14.0701,14.2779] | 18.812 |
| current-strata | `occ16-clustered-skewed-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 13.2175 | [13.1593,13.3719] | 10.907 |
| current-strata | `occ16-clustered-skewed-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 13.1937 | [13.1811,13.2302] | 10.894 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 13.0705 | [12.9891,13.1988] | 18.923 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.8636 | [12.7463,13.1009] | 18.812 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 12.3244 | [12.2749,12.5069] | 10.092 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.2664 | [12.3033,12.3288] | 9.814 |
| current-strata | `occ2-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 12.2547 | [12.1547,12.2719] | 8.727 |
| current-strata | `occ2-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.1343 | [12.0876,12.2423] | 8.645 |
| current-strata | `occ4-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 12.1534 | [11.9544,12.1743] | 8.780 |
| current-strata | `occ4-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.0674 | [12.0383,12.1629] | 8.697 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 11.8644 | [11.8848,11.9381] | 9.098 |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 11.7557 | [11.7570,11.9275] | 9.009 |

The worst cell moved from the spill **present-hit** path to the spill
**absent-miss** path. The new worst present-hit cells are `occ8-offset-beyond-direct-present-hit-*`
(14.58/14.61×, ~10.9 ns/op); `occ16` present-hit is 12.27/12.32× at 9.81/10.09 ns/op,
matching the ~9.9 ns/op micro measurement.

### 6 · Spatial failures (10 rows, none passing)

| cell | selected | incumbent | ratio | interval | concurrency | address_face | basis |
|---|---|---|---|---|---|---|---|
| `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 0.7953 | [0.7830,0.8718] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.6974 | [2.5125,2.9699] | MISMATCH | unmatched | serial |
| `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1167 | [1.0911,1.1674] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `foldhash_par16 w16` | 2.6115 | [2.5237,2.8098] | matched | matched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `fxhash_par16 w16` | 3.1125 | [2.7908,3.4410] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.8653 | [2.5369,3.1616] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 23.0538 | [22.0638,23.1265] | MISMATCH | unmatched | serial |
| `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.0689 | [1.0193,1.0728] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `foldhash_par16 w16` | 5.6274 | [5.3468,6.1887] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `fxhash_par16 w16` | 6.0770 | [5.8106,6.2807] | matched | matched | parallel |

`spatial/n100000` vs `count_par` is under the ratio limit (0.7953) but fails the
address-face gate. The two `get_each` rows are counted as concurrency-unmatched
(2); the four `count_par`/`hashbrown_rayon` rows are the address-face-unmatched (4).
The four `foldhash_par16`/`fxhash_par16` rows are ratio failures.

### 7 · Best passing pairs (top-10 by ratio ascending)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ8-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0757 | [0.0733,0.0784] | 3.888 |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0784 | [0.0784,0.0785] | 3.885 |
| current-strata | `occ8-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0785 | [0.0781,0.0785] | 3.897 |
| current-strata | `occ16-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0787 | [0.0786,0.0804] | 3.897 |
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0788 | [0.0786,0.0832] | 3.906 |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0790 | [0.0785,0.0792] | 3.911 |
| current-strata | `occ2-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0790 | [0.0784,0.0819] | 3.889 |
| current-strata | `occ4-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0820 | [0.0807,0.0825] | 3.890 |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0821 | [0.0788,0.0823] | 3.887 |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0824 | [0.0820,0.0828] | 3.892 |

All ten are `existing-update` on the resident direct-index path, as in the post-spill
receipt.

### 8 · Post-spill vs post-correctness on the previously-worst cells

`pre` is the post-spill receipt (its §5/§6 "post" columns); `post` is this run. Same
cell, incumbent and worker width. `ns/op = sel_median_ns / 4096`.

#### Spill present-hit

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table w1` | 30.9208 | 24.69 | 12.3244 | 10.092 | [12.2749,12.5069] |
| `occ16-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table w1` | 30.7018 | 24.74 | 12.2664 | 9.814 | [12.3033,12.3288] |
| `occ16-seeded-random-present-hit-replacement` | `hashtrinity_table w1` | 22.5222 | 24.68 | 9.6735 | 10.200 | [9.4370,9.8420] |
| `occ16-seeded-random-present-hit-wrapping-add` | `hashtrinity_table w1` | 22.3041 | 24.69 | 9.7360 | 10.280 | [9.7871,9.8435] |

The tag fingerprint cut the spill present-hit selected cost from ~24.7 to ~9.8–10.3
ns/op (ratio ~30.9 → ~12.3; the incumbent `hashtrinity_table` side is stable at
~0.80–0.82 ns/op in both runs).

#### Spill absent-miss

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash w1` | 23.5900 | 19.68 | 20.4682 | 16.161 | [20.0871,20.5922] |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 23.7463 | 19.68 | 18.2012 | 15.993 | [18.1236,18.5058] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash w1` | 17.4845 | 17.61 | 19.1104 | 18.812 | [18.5575,19.3316] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 18.1126 | 17.67 | 19.8637 | 18.923 | [19.1943,20.2727] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table w1` | 9.5818 | 17.61 | 12.8636 | 18.812 | [12.7463,13.1009] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table w1` | 9.6489 | 17.67 | 13.0705 | 18.923 | [12.9891,13.1988] |

The `seeded-random` absent-miss selected cost fell ~19.7 → ~16.0–16.2 ns/op; the
`offset-beyond-direct` absent-miss selected cost **rose** ~17.6 → ~18.8–18.9 ns/op.
The `hashtrinity_table` ratios rose more than the selected cost alone because the
incumbent side measured faster this run (derived ~1.84 → ~1.46 ns/op on that cell);
that is a measurement-attribution fact, not a change to the pinned incumbent.

#### Growth-insert

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table w1` | 1.5275 | 13.31 | 1.5364 | 12.561 | [1.5075,1.5367] |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table w1` | 1.4905 | 12.84 | 1.5227 | 12.571 | [1.5261,1.5443] |

Essentially unchanged; still the only growth-insert pairs below ~1.6×, and still
failing the `<= 1.0` interval contract.

#### Resident absent-miss (named in the post-spill §5 note)

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-absent-miss-replacement` | `hashtrinity_table w1` | 4.8359 | 2.65 | 4.9286 | 2.645 | [4.7962,4.9293] |
| `occ16-direct-index-resident-absent-miss-replacement` | `hashbrown_foldhash w1` | 2.1052 | 2.65 | 1.9998 | 2.645 | [1.9202,2.0638] |

### 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

| incumbent | pass | fail | n/a | info | post-spill pass/fail |
|---|---|---|---|---|---|
| `hashtrinity_table` | 28 | 132 | 2 | 0 | 28 / 132 |
| `hashbrown_foldhash` | 22 | 138 | 2 | 0 | 14 / 146 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 | 0 / 160 |
| `std_siphash` | 92 | 68 | 2 | 0 | 86 / 74 |
| `xxhash_xxh3` | 144 | 16 | 2 | 0 | 152 / 8 |
| `xxhash_xxh64` | 42 | 118 | 2 | 0 | 32 / 128 |
| `wyhash` | 13 | 147 | 2 | 0 | 12 / 148 |
| `ahash` | 34 | 126 | 2 | 0 | 26 / 134 |
| `fnv` | 46 | 114 | 2 | 0 | 37 / 123 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `blake3` | 0 | 0 | 0 | 162 | 0 / 0 |
| `sha256` | 0 | 0 | 0 | 162 | 0 / 0 |

### 10 · Controls

55 control observations, 0 green (`observed=red`): the 19 acceptance/registration
controls (`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`), `placement-only-address`,
`fabricated-current-cell`, and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red; some print twice).

### 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above; the derived
  `ns/op = median_ns / 4096`.
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`. Extraction of `is_intact` is blocked separately by Aeneas and
  is not exercised here.
- **Grade**: `measured` for the gate result; the digests and log hash are identity
  receipts. The 55 controls establish the harness rejects corrupted observations,
  not that the table is correct.

### 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair table is retained only at the
  `/tmp` log path with the sha256 in §2; durable retention is not included here.
  The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and
  were not captured in table mode.
- Next bounded task: attribute the spill **absent-miss** probe cost — the new worst
  cells `occ16-seeded-random-absent-miss-*` (20.47/18.20×, ~16.0–16.2 ns/op vs
  `hashbrown_fxhash`) and `occ16-offset-beyond-direct-absent-miss-*`
  (19.86/19.11×, ~18.8–18.9 ns/op) — to a named `src/table.rs`/`src/table/scan.rs`
  region, with a focused before/after measurement on the same 4096-op schedule.
  Do not weaken the `<= 1.0` contract.

## REG-1 performance remeasurement, post summary-mask (APEIRON-PERF-REMEASURE-T5)

Fresh strict-gate measurement on the current source, after the `least_set_lane`
halving search in `spill_read` (`src/table/scan.rs`), the power-of-two summary
mask fast path (`src/table/state.rs`) and the single-key-qualification fix in
`src/table/parallel.rs` (`get_each`). The prior receipt
([`perf-ledger.md`](perf-ledger.md))
binds `src/table.rs` sha256 `5f260801…` and HEADLINE `pass=500 fail=950`; the
executing source is now `e568d0c3…`. This receipt supersedes the post-scatter
numbers for the current identity only.

Three full passes ran on one build. Pass 1 (the command below) is the primary
receipt; passes 2 and 3 are the reproducibility control in §4a. The three
HEADLINEs are `pass=561`, `pass=540`, `pass=547`.

### 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `e568d0c318801304d17ad304a1f719d509de28b35b33764ad6980619e3e4856d` |
| `src/table/state.rs` | `8dad06425b7ae68b47657e97d7f31ece917b0dd87b7c520d7e1b4f321bf64f8a` |
| `src/table/scan.rs` | `ec633507a8ad456a7cb11553f559cf1abc08d3d4984a118e145eb5aa1ad91e97` |
| `src/table/parallel.rs` | `bd843ddcf5fb8074736a52ffbca4fa0bc1bf872839d770100e54c81f59ab1290` |
| `src/table/partition.rs` | `7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64` |
| `proof/ApeironWord.lean` | `9af54e7d2467a5a19ebccfd7fd70691072a17fda76e78e2366c3832675749d3e` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `bf5d26106064e944b620e9922a2e30004b9d24ef8dffb0b8698c1f678fd91654` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the post-scatter receipt: `src/table.rs` (`5f260801…` →
`e568d0c3…`), `src/table/state.rs` (`79983466…` → `8dad0642…`), `src/table/scan.rs`
(`698b1eca…` → `ec633507…`), `src/table/parallel.rs` (`b22650c1…` → `bd843ddc…`),
`Cargo.toml` (`9d445226…` → `bf5d2610…`). `src/table/partition.rs` (`7cf47244…`)
is unchanged. The `Cargo.toml` delta adds the off-by-default `read-stats` feature
(§11). The reg1 harness sources and `Cargo.lock` are byte-identical to the
post-scatter receipt. The run writes none of these files. `src/table.rs` changed
beyond the three named edits; the receipt binds its digest and does not attribute
the delta.

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| CPU governor / turbo | `powersave`; `intel_pstate/no_turbo=0` |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run 1 log | `/tmp/opencode/reg1-full-post-mask-20260920-080201.log`, sha256 `157b0108470b1f163c4af858a1839b527a12429f834528fe2f1faaef692eb618`, 2690 lines |
| run 2 log | `/tmp/opencode/reg1-full-post-mask-20260920-080329.log`, sha256 `2848212bab6f9af51b9fe7a57e5f089ffd93a3b42d22e1e6d3d7dcd80d47c6f8`, 2690 lines |
| run 3 log | `/tmp/opencode/reg1-full-post-mask-20260920-080535.log`, sha256 `2e2f4f0e9e360e5b7bb1c30e1669d92d5f55ec91ebdd6891eb36579c97de0bb7`, 2690 lines |
| build log | `/tmp/opencode/reg1-build-post-mask-20260920-080157.log`, sha256 `bdaeb7034df08ce2bb2245b7eabab478ab7d49488257a1a30b7056f5b4e51f0e`, 30 lines |

No concurrent cargo/Lean/Kani/coverage build ran; `pgrep` found none before run 1.
A `btrfs-cleaner` kernel thread and a desktop session were resident, so the host
was not fully idle: load average `1.00/1.60/1.67` before run 1, `1.85/1.95/1.80`
after run 3.

### 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.42; user 11.55 s; max RSS 554484 KiB; "Finished in 2.37s"

# 2. strict full gate, pass 1 (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-post-mask-20260920-080201.log 2>&1
# exit 1 (strict acceptance failure); wall 0:01:25.97; user 173.78 s; system 8.28 s;
# 211% CPU; max RSS 782996 KiB. timeout 3600 NOT reached.

# 3. strict full gate, pass 2 (reproducibility control)
#    same command -> /tmp/opencode/reg1-full-post-mask-20260920-080329.log
# exit 1; wall 0:01:25.79; user 171.02 s; system 8.67 s; 209% CPU; max RSS 868028 KiB.

# 4. strict full gate, pass 3 (reproducibility control)
#    same command -> /tmp/opencode/reg1-full-post-mask-20260920-080535.log
# exit 1; wall 0:01:25.74; user 174.79 s; system 8.56 s; 213% CPU; max RSS 738904 KiB.
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

### 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=561 fail=889 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 883 are ratio failures. Against the
post-scatter receipt (`pass=500 fail=950`, same `registered`/`load_bearing`/
`applicable`): `pass` rose 500 → 561, `fail` fell 950 → 889 on pass 1.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over
one deterministic serial schedule (`STRATA_OPS = 4096` registered ops,
`campaigns=3 warmup=2 samples=15`), interval = min/max campaign ratio; acceptance
requires interval max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows.
Spatial rows carry raw elapsed wall ns for the packed operation, not ns/op; their
ratio is `selected_wall_ns / incumbent_wall_ns`. The pooled point ratio can fall
outside the campaign min/max interval; acceptance reads the interval max.

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

#### 4a · Reproducibility (passes 2 and 3)

| pass | pass | fail | status flips vs pass 1 |
|---|---|---|---|
| 1 | 561 | 889 | — |
| 2 | 540 | 910 | 47 |
| 3 | 547 | 903 | 54 |

The three-run `pass` range is 540–561 (mean 549.3), 21 pairs wide (1.4% of
`applicable`). The post-scatter control was 1 pair wide (500/501); run-to-run
variance is larger on this identity. Every post-mask run still exceeds every
post-scatter run by at least 39 pairs. The selected-side median moved more than
20% on the four `occ1-*` present-hit cells, marked `†` in §5:

| cell | pass 1 sel ns/op | pass 2 sel ns/op |
|---|---|---|
| `occ1-direct-index-resident-present-hit-replacement` | 1.903 | 3.363 |
| `occ1-direct-index-resident-present-hit-wrapping-add` | 1.768 | 2.289 |
| `occ1-offset-beyond-direct-present-hit-replacement` | 7.192 | 12.166 |
| `occ1-offset-beyond-direct-present-hit-wrapping-add` | 7.346 | 10.318 |

42 of the 47 pass-1/pass-2 flips are pairs whose ratio sits within ~7% of 1.0
(0.93–1.07), where absolute noise changes the verdict. The 5 far flips are
`occ1-direct-index-resident-present-hit-*` vs `xxhash_xxh64` (0.75/0.72 pass →
1.13/0.90 fail), `occ1-clustered-skewed-present-hit-replacement` vs
`xxhash_xxh3` (0.33 pass → 1.07 fail), and
`occ16-direct-index-resident-existing-update-*` vs `hashtrinity_table`
(1.73/1.76 fail → 0.66/0.66 pass). The `occ1` present-hit selected cost is
order-unstable, so its rows are not a stable ranking signal.

### 5 · Worst failing pairs (top-20 by interval max, pass 1)

`ns/op` is `sel_median_ns / 4096` for serial rows; spatial carries raw ms. `†`
marks an order-unstable selected cell (§4a).

| family | cell | selected | incumbent | ratio | interval | sel |
|---|---|---|---|---|---|---|
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 17.5729 | [17.4485,17.7193] | 276.930 ms |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `apeiron_table w1` | `hashtrinity_table w1` | 10.0696 | [9.9751,10.1370] | 7.346 ns/op |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashtrinity_table w1` | 10.0365 | [10.0124,10.0526] | 7.192 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 10.0187 | [9.9006,9.9494] | 7.328 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 9.9973 | [9.8944,9.9322] | 7.317 ns/op |
| current-strata | `occ2-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 9.7508 | [9.6861,9.8460] | 7.004 ns/op |
| current-strata | `occ2-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 9.7848 | [9.7247,9.8134] | 7.028 ns/op |
| current-strata | `occ4-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.9191 | [8.9108,8.9334] | 6.672 ns/op |
| current-strata | `occ4-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.8887 | [8.8754,8.9129] | 6.651 ns/op |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.7307 | [8.7173,8.8315] | 6.586 ns/op |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.7487 | [8.7277,8.7586] | 6.570 ns/op |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.2160 | [8.1221,8.2785] | 6.908 ns/op |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.2200 | [8.0886,8.2396] | 6.887 ns/op |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.9600 | [6.1405,6.9713] | 7.346 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.9541 | [6.7226,6.9661] | 7.317 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.9498 | [5.8053,6.9553] | 7.328 ns/op |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.8203 | [5.8779,6.9211] | 7.192 ns/op |
| current-strata | `occ1-direct-index-resident-admitted-fresh-insert-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 6.7675 | [6.7271,6.8236] | 1.698 ns/op |
| current-strata | `occ1-direct-index-resident-admitted-fresh-insert-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 6.7678 | [6.7055,6.8193] | 1.694 ns/op |
| current-strata | `occ2-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.7007 | [6.6827,6.7615] | 7.023 ns/op |

The worst **stable** current-strata cells are the spill present-hit rows at
`occ2`/`occ4`/`occ8`/`occ16` vs `hashtrinity_table` (8.22–9.78×, ~6.6–7.0 ns/op),
reproduced in passes 2 and 3. The two
`occ1-direct-index-resident-admitted-fresh-insert-*` rows (6.77×) are not new:
the post-scatter receipt measured 6.86/6.92×, so they only entered the top-20
because the `occ1` present-hit rows above them reordered. The spill absent-miss
cells that led the post-correctness top-20 remain off this list.

### 6 · Spatial rows (raw wall ns; all 10 rows fail)

| cell | selected | incumbent | ratio | interval | concurrency | address_face |
|---|---|---|---|---|---|---|
| `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 1.3786 | [1.0584,1.8408] | matched | unmatched |
| `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.3105 | [2.4786,2.7020] | MISMATCH | unmatched |
| `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.2626 | [1.0639,1.3597] | matched | unmatched |
| `spatial/n100000` | `apeiron_table w16` | `foldhash_par16 w16` | 2.8983 | [2.6416,3.0252] | matched | matched |
| `spatial/n100000` | `apeiron_table w16` | `fxhash_par16 w16` | 2.8851 | [2.7775,3.2596] | matched | matched |
| `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.6677 | [2.5128,2.6803] | matched | unmatched |
| `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 17.5729 | [17.4485,17.7193] | MISMATCH | unmatched |
| `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1487 | [1.0580,1.1652] | matched | unmatched |
| `spatial/n1000000` | `apeiron_table w16` | `foldhash_par16 w16` | 5.8085 | [5.4319,6.2537] | matched | matched |
| `spatial/n1000000` | `apeiron_table w16` | `fxhash_par16 w16` | 5.3499 | [5.2835,5.6112] | matched | matched |

The two `get_each` rows are the concurrency-unmatched (2); the four
`count_par`/`hashbrown_rayon` rows are the address-face-unmatched (4); the four
`foldhash_par16`/`fxhash_par16` rows are ratio failures. The `n100000`
`count_par` ratio is unstable across passes (1.379/0.967/1.025) and is
address-face-unmatched regardless of ratio. The `n1000000` `hashbrown_rayon`
ratio is likewise unstable (1.149/0.980/1.094). The `get_each` selected wall
cost fell 308.9 → 276.9 ms (pass 1; 273.4/269.0 in passes 2/3), consistent with
the `get_each` fix, but the row is concurrency-MISMATCH and therefore not
applicable.

### 7 · Best passing pairs (top-10 by ratio ascending, pass 1)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0595 | [0.0594,0.0602] | 3.006 |
| current-strata | `occ8-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0604 | [0.0603,0.0606] | 1.329 |
| current-strata | `occ2-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0606 | [0.0606,0.0610] | 1.329 |
| current-strata | `occ2-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0607 | [0.0606,0.0610] | 1.332 |
| current-strata | `occ16-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0607 | [0.0606,0.0608] | 1.332 |
| current-strata | `occ8-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0608 | [0.0605,0.0642] | 1.334 |
| current-strata | `occ16-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0608 | [0.0607,0.0610] | 1.335 |
| current-strata | `occ4-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0609 | [0.0609,0.0610] | 1.333 |
| current-strata | `occ4-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0609 | [0.0605,0.0612] | 1.331 |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0610 | [0.0610,0.0612] | 3.025 |

All ten are resident direct-index paths against `xxhash_xxh3`. The resident
absent-miss selected cost is ~1.33 ns/op, unchanged from post-scatter.

### 8 · Post-scatter vs post-mask on the previously-worst cells

`pre` is the post-scatter receipt; `post` is pass 1; `post 2` is pass 2. Same
cell, incumbent and worker width. `ns/op = sel_median_ns / 4096`. Ratios compare
the selected side to a *re-measured* incumbent, so a ratio move is not
attributable to the source alone; the incumbent ns/op is given where it moved.

#### Spill present-hit

| cell | incumbent | pre ratio | pre sel | post ratio | post sel | post interval | post 2 sel |
|---|---|---|---|---|---|---|---|
| `occ16-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 10.2197 | 8.119 | 8.2200 | 6.887 | [8.0886,8.2396] | 6.544 |
| `occ16-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 10.0217 | 8.133 | 8.2160 | 6.908 | [8.1221,8.2785] | 6.548 |
| `occ8-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 8.8709 | 6.675 | 8.7307 | 6.586 | [8.7173,8.8315] | 6.586 |
| `occ8-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 8.8014 | 6.674 | 8.7487 | 6.570 | [8.7277,8.7586] | 6.572 |
| `occ4-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 8.4284 | 6.282 | 8.9191 | 6.672 | [8.9108,8.9334] | 6.677 |
| `occ4-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 8.4231 | 6.280 | 8.8887 | 6.651 | [8.8754,8.9129] | 6.707 |
| `occ2-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 8.6316 | 6.172 | 9.7848 | 7.028 | [9.7247,9.8134] | 7.054 |
| `occ2-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 8.6612 | 6.185 | 9.7508 | 7.004 | [9.6861,9.8460] | 6.999 |
| `occ1-offset-beyond-direct-present-hit-replacement` † | `hashtrinity_table` | 8.7629 | 6.450 | 10.0365 | 7.192 | [10.0124,10.0526] | 12.166 |
| `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `hashtrinity_table` | 8.5953 | 6.155 | 10.0696 | 7.346 | [9.9751,10.1370] | 10.318 |

The selected cost improves only at `occ16` (8.12 → 6.89 ns/op, −15.2%); `occ8` is
flat (6.67 → 6.59); `occ4`, `occ2` and `occ1` regress (6.28 → 6.67, 6.17 → 7.03,
6.45 → 7.19). The incumbent `hashtrinity_table` side is stable at 0.72–0.84
ns/op, so the ratio follows the selected side. The micro-benchmark's present-hit
−15.6% holds on this schedule only at `occ16`; the low-occupancy cells are not
improved. The `occ1` rows are order-unstable (§4a).

#### Spill absent-miss

| cell | incumbent | pre ratio | pre sel | post ratio | post sel | post interval | post 2 sel |
|---|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash` | 4.3922 | 3.675 | 3.3602 | 2.794 | [3.3534,3.3810] | 2.789 |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash` | 4.4443 | 3.680 | 3.3564 | 2.791 | [3.3418,3.3802] | 2.789 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash` | 3.3645 | 3.353 | 2.5141 | 2.476 | [2.5063,2.5226] | 2.474 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash` | 3.4118 | 3.354 | 2.4972 | 2.476 | [2.4957,2.5021] | 2.474 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table` | 2.4425 | 3.353 | 1.7970 | 2.476 | [1.7873,1.8043] | 2.474 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table` | 2.4346 | 3.354 | 1.7987 | 2.476 | [1.7898,1.8042] | 2.474 |

The selected cost fell 3.675 → 2.794 (−24.0%) and 3.353 → 2.476 (−26.2%) ns/op,
reproducing in passes 2 and 3. This matches the −23.9% absent-miss micro claim.
The `hashbrown_fxhash` incumbent is stable at ~0.83–0.99 ns/op.

#### Growth-insert

| cell | incumbent | pre ratio | pre sel | pre inc | post ratio | post sel | post inc | post 2 ratio |
|---|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table` | 2.3943 | 12.173 | 5.084 | 1.4629 | 11.880 | 8.121 | 2.1049 |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table` | 2.3358 | 12.051 | 5.159 | 1.4575 | 11.881 | 8.152 | 2.0941 |

The selected cost holds (12.17 → 11.88 ns/op). The ratio move 2.39 → 1.46 is
incumbent-side: the re-measured `hashtrinity_table` rose 5.08 → 8.12 ns/op on
pass 1 and fell to ~5.6 on pass 2 (ratio 2.10). This is cross-session incumbent
drift, not a selected-side change.

#### Resident absent-miss

| cell | incumbent | pre ratio | pre sel | post ratio | post sel | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-absent-miss-replacement` | `hashtrinity_table` | 2.4699 | 1.344 | 2.4757 | 1.332 | [2.4721,2.4749] | 2.4660 |
| `occ16-direct-index-resident-absent-miss-replacement` | `hashbrown_foldhash` | 0.9616 | 1.344 | 0.9416 | 1.332 | [0.9329,0.9501] | 1.0395 |
| `occ16-direct-index-resident-absent-miss-wrapping-add` | `hashbrown_foldhash` | 0.9859 | 1.352 | 0.9468 | 1.335 | [0.9355,0.9511] | 1.0286 |

The selected cost is unchanged (~1.33 ns/op). The pair against
`hashbrown_foldhash` passes on pass 1 (0.942/0.947) but fails on pass 2
(1.040/1.029): the two sides are within ~5%, so this pair is not a stable signal.

#### Spatial

Raw elapsed wall ns (not ns/op); `sel`/`inc` are pass-1 wall ns.

| cell | incumbent | pre ratio | post ratio | post sel | post inc | post 2 ratio |
|---|---|---|---|---|---|---|
| `spatial/n100000` | `hashtrinity_count_par w24` | 1.2654 | 1.3786 | 4.301 ms | 3.119 ms | 0.9669 |
| `spatial/n100000` | `hashtrinity_get_each w24` | 2.6256 | 2.3105 | 8.947 ms | 3.872 ms | 2.0179 |
| `spatial/n100000` | `hashbrown_rayon w32` | 1.2810 | 1.2626 | 3.915 ms | 3.101 ms | 1.3093 |
| `spatial/n100000` | `foldhash_par16 w16` | 2.9151 | 2.8983 | 3.179 ms | 1.097 ms | 2.8091 |
| `spatial/n100000` | `fxhash_par16 w16` | 3.0204 | 2.8851 | 2.881 ms | 0.999 ms | 2.8763 |
| `spatial/n1000000` | `hashtrinity_count_par w24` | 2.7982 | 2.6677 | 41.701 ms | 15.632 ms | 2.5032 |
| `spatial/n1000000` | `hashtrinity_get_each w24` | 20.0474 | 17.5729 | 276.930 ms | 15.759 ms | 17.7872 |
| `spatial/n1000000` | `hashbrown_rayon w32` | 1.1893 | 1.1487 | 46.803 ms | 40.745 ms | 0.9800 |
| `spatial/n1000000` | `foldhash_par16 w16` | 6.2851 | 5.8085 | 49.585 ms | 8.537 ms | 5.3684 |
| `spatial/n1000000` | `fxhash_par16 w16` | 5.7893 | 5.3499 | 45.195 ms | 8.448 ms | 5.4976 |

The prior receipt stated "~1104 ns" for the `n100000` `count_par` selected wall;
that figure is `sel_median_ns / 4096` (4522358 / 4096 = 1104.1), not the raw
elapsed wall ns (4.52 ms). This receipt reports the raw measurement. Ratios are
mixed and within ~±25% of post-scatter; the `get_each` rows are
concurrency-MISMATCH and the `count_par`/`hashbrown_rayon` rows are
address-face-unmatched, so no spatial row is applicable.

### 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

Pass 1:

| incumbent | pass | fail | n/a | info | post-scatter pass/fail |
|---|---|---|---|---|---|
| `hashtrinity_table` | 32 | 128 | 2 | 0 | 28 / 132 |
| `hashbrown_foldhash` | 26 | 134 | 2 | 0 | 25 / 135 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 | 0 / 160 |
| `std_siphash` | 128 | 32 | 2 | 0 | 123 / 37 |
| `xxhash_xxh3` | 156 | 4 | 2 | 0 | 152 / 8 |
| `xxhash_xxh64` | 73 | 87 | 2 | 0 | 54 / 106 |
| `wyhash` | 20 | 140 | 2 | 0 | 20 / 140 |
| `ahash` | 47 | 113 | 2 | 0 | 43 / 117 |
| `fnv` | 79 | 81 | 2 | 0 | 55 / 105 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `blake3` | 0 | 0 | 0 | 162 | 0 / 0 |
| `sha256` | 0 | 0 | 0 | 162 | 0 / 0 |

Every incumbent except `hashbrown_fxhash` and `wyhash` gained passes;
`hashbrown_foldhash` gained one, `hashtrinity_table` four, `fnv` 24.

### 10 · Controls

55 control observations, 0 green (`observed=red`), identical in all three passes
and to the post-scatter receipt: the 19 acceptance/registration controls
(`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`), `placement-only-address` (×2),
`fabricated-current-cell` (×2), and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red).

### 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above; the derived
  `ns/op = median_ns / 4096`; the three-pass reproducibility check (47–57 status
  flips, 1.8–2.2%).
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`. The extraction identity (`ApeironWord.lean` `9af54e7d…`) is
  recorded but its freshness was not re-run here.
- **Timed path is counter-free**: `Cargo.toml` adds the `read-stats` feature, off
  by default; the `--release --locked --example reg1` invocation enables no
  feature, so the `thread_local` counters in `src/table.rs` are compiled out of
  the timed path.
- **Grade**: `measured` for the gate result; the digests and log hashes are
  identity receipts. The 55 controls establish the harness rejects corrupted
  observations, not that the table is correct. The three-pass comparison bounds,
  but does not eliminate, run-to-run variance; the host was not fully idle (§2)
  and the 21-pair spread exceeds the post-scatter control's 1-pair spread.
- **Cell semantics**: no cell showed a semantically wrong value (wrong operation,
  wrong address face, wrong status class). The `occ1` present-hit and the
  near-1.0 ratio swings are timing/ordering effects, not semantics; no Sol
  escalation triggered. The present-hit micro claim (−15.6%) is not uniform over
  occupancy on this schedule; that is a measurement result, not a semantic error.

### 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair tables are retained only at the
  `/tmp` log paths with their sha256 in §2; durable retention is not included
  here. The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only
  and were not captured in table mode. The `occ1` present-hit order-instability is
  characterized but not root-caused; it affects four cells and several near-1.0
  pairs. The present-hit regression at `occ2`/`occ4`/`occ1` (ratio 8.63 → 9.78,
  8.43 → 8.92, 8.76 → 10.04 vs `hashtrinity_table`) is measured but not
  attributed.
- Next bounded task: attribute the worst **stable** current-strata ratio — the
  spill **present-hit** path at `occ2-offset-beyond-direct-present-hit-*` vs
  `hashtrinity_table` (~9.78×, ~7.0 ns/op) and the low-occupancy `occ4`/`occ1`
  cells — to a named `src/table.rs`/`src/table/scan.rs`/`src/table/state.rs`
  region, with a focused before/after measurement on the same 4096-op schedule.
  Determine whether the power-of-two summary mask fast path is bypassed at low
  occupancy. Secondary: the `occ1` present-hit order-instability. Do not weaken
  the `<= 1.0` contract.

### Arc Block

```text
SCOPE:      docs/evidence/perf-ledger.md (new receipt only);
            src/*, tests/*, proof/*, examples/* untouched
REGISTER:   measurement; no algebra/refinement/state obligation
CORPUS:     no canonical-corpus match (performance receipt); ApeironWord.lean digest recorded, freshness not re-run
REFERENCE:  docs/evidence/perf-ledger.md; examples/reg1/acceptance.rs, performance.rs, strata.rs
GATE:       cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
            cwd /home/socol/Workspace/apeiron; observed red exit 1, HEADLINE pass=561 fail=889
GRADE:      measured (three passes); digests and log hashes are identity receipts
```

## REG-1 performance remeasurement, post stable scatter (APEIRON-PERF-REMEASURE-T4)

Fresh strict-gate measurement on the current source, after the spill tag
fingerprint (`tag = TAG_LIVE | ((place >> 8) as u8 & 0x7f)`), the specialized
read path (single-exit `spill_read`, inlined resident), the totalized summary,
and the parallel stable scatter. The prior receipt
([`perf-ledger.md`](perf-ledger.md))
binds `src/table.rs` sha256 `00c3c7f8…`; the executing source is now `5f260801…`.
The prior numbers are stale for this identity; this receipt supersedes them for
the current identity only.

Two full passes ran on the same build. Pass 1 (the command below) is the primary
receipt; pass 2 is the reproducibility control in §4a. The two HEADLINEs differ by
one pair (`pass=500 fail=950` vs `pass=501 fail=949`).

### 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `5f2608010137b16b195baad62cb5214bcadeeea13a9df4da39e19f4bf543a61e` |
| `src/table/state.rs` | `799834665532bd4361ecd7fbd3c1497fffaef42a7044139de9e91fdd29e1f3e8` |
| `src/table/partition.rs` | `7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64` |
| `src/table/parallel.rs` | `b22650c1c6db76a7865e782fc0fc56174d3b8b6ac2c7eaccc0af087815df193f` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/resources.rs` | `a1d6e242c912706752375027c21b255677830ec74254d89fc8d5dd1324c06009` |
| `proof/ApeironWord.lean` | `4c7bb34f8e4926daacf349ba828d5cf0d24dc725e3b96ceec3fda3fe48ca8bf6` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the post-correctness receipt: `src/table.rs`
(`00c3c7f8…` → `5f260801…`), `src/table/partition.rs` (`7207a4e9…` →
`7cf47244…`), `src/table/parallel.rs` (`66f0ba73…` → `b22650c1…`); `state.rs`
was absent from the post-correctness binding table. `scan.rs` (`698b1eca…`) and
`resources.rs` (`a1d6e242…`) are unchanged. The reg1 harness sources (`main.rs`,
`performance.rs`, `acceptance.rs`, `strata.rs`, `matrix.rs`, `matrix.json`,
`Cargo.toml`, `Cargo.lock`) are byte-identical to the post-correctness receipt.
The run writes none of these files.

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| CPU governor / turbo | `powersave`; `intel_pstate/no_turbo=0` |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run 1 log | `/tmp/opencode/reg1-full-postscatter-20260919-210424.log`, sha256 `3916a46821de226202e143f8f5ab2bb0fb73e960cc44d46de36420c61cd79ce3`, 2690 lines |
| run 2 log | `/tmp/opencode/reg1-full-postscatter-20260919-210709.log`, sha256 `7e1ffe7255fa2c49ac8042ee12d1dd438e7c4f24ea9a06e871a9c8c44ea10182`, 2690 lines |
| build log | `/tmp/opencode/reg1-build-postscatter-20260919-210419.log`, sha256 `10c8c46765848a03433fe7129b5da86d7c32510dc4e98a6c495a1f602fde88eb`, 30 lines |

No concurrent build ran (no cargo/Lean/Kani/coverage). A desktop session
(Hyprland, Discord, ChatGPT, sunshine) was resident, so the host was not fully
idle: load average `1.61/1.27/1.15` before run 1, `1.54/1.32/1.18` after run 1,
`4.17/2.40/1.60` after run 2. This is higher than the post-correctness receipt's
`0.65/0.86/1.20`; the effect is quantified in §4a.

### 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.30; user 11.22 s; max RSS 565788 KiB

# 2. strict full gate, pass 1 (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-postscatter-20260919-210424.log 2>&1
# exit 1 (strict acceptance failure); wall 0:01:28.63; user 189.04 s; system 8.07 s;
# 222% CPU; max RSS 821704 KiB. timeout 3600 NOT reached; smoke fallback not run.

# 3. strict full gate, pass 2 (reproducibility control)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-postscatter-20260919-210709.log 2>&1
# exit 1; wall 0:01:28.54; user 186.54 s; system 8.35 s; 220% CPU; max RSS 757900 KiB.
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

### 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=500 fail=950 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 944 are ratio failures. Against the
post-correctness receipt (`pass=421 fail=1029`, same `registered`/`load_bearing`/
`applicable`): `pass` rose 421 → 500, `fail` fell 1029 → 950.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).
The pooled point ratio can fall outside the campaign min/max interval (observed on
several cells, both directions); acceptance reads the interval max.

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

#### 4a · Reproducibility (pass 2)

Pass 2 HEADLINE: `pass=501 fail=949 missing=0 concurrency_unmatched=2
address_face_unmatched=4`. 29 of 2592 pairs (1.1%) changed status between the two
passes. The selected-side median moved more than 20% on exactly three cells:

| cell | pass 1 sel ns/op | pass 2 sel ns/op |
|---|---|---|
| `occ1-offset-beyond-direct-present-hit-replacement` | 6.450 | 10.339 |
| `occ1-seeded-random-present-hit-replacement` | 10.895 | 6.083 |
| `occ1-seeded-random-present-hit-wrapping-add` | 9.773 | 6.177 |

The `occ1` present-hit selected cost is order-unstable: the offset-beyond and
seeded-random replacement cells exchange ~6.2 and ~10.3 ns/op between passes. Rows
carrying these cells are marked `†` in §5; their ratios are not a stable ranking
signal. The largest ratio swings overall are against `xxhash_xxh3`, whose incumbent
side sits near 0.7–0.8 ns/op so small absolute noise becomes a large ratio change;
the selected side is stable there.

### 5 · Worst failing pairs (top-20 by interval max, pass 1)

`ns/op` is `sel_median_ns / 4096`. `†` marks a row whose selected cell is
order-unstable between passes (§4a).

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.0474 | [19.8189,20.8308] | — |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.1200 | [5.8410,12.8527] | 6.450 |
| current-strata | `occ1-seeded-random-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 10.3349 | [8.9431,11.4965] | 10.895 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_foldhash w1` | 5.2724 | [5.0395,11.0525] | 6.450 |
| current-strata | `occ1-seeded-random-present-hit-wrapping-add` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 9.3093 | [8.1506,10.7577] | 9.773 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 10.2197 | [10.1991,10.2151] | 8.119 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 10.0217 | [9.8928,10.1629] | 8.133 |
| current-strata | `occ1-seeded-random-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_foldhash w1` | 8.5951 | [7.3920,9.5964] | 10.895 |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.8014 | [8.7901,8.9352] | 6.674 |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.8709 | [8.8691,8.9128] | 6.675 |
| current-strata | `occ1-seeded-random-present-hit-wrapping-add` † | `apeiron_table w1` | `hashbrown_foldhash w1` | 7.6190 | [5.3492,8.8575] | 9.773 |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `apeiron_table w1` | `hashtrinity_table w1` | 8.5953 | [8.5658,8.8288] | 6.155 |
| current-strata | `occ1-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.7713 | [8.7176,8.8262] | 6.733 |
| current-strata | `occ2-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.6612 | [8.6271,8.6680] | 6.185 |
| current-strata | `occ2-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.6316 | [8.6266,8.6557] | 6.172 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashtrinity_table w1` | 8.7629 | [8.5870,8.6119] | 6.450 |
| current-strata | `occ1-clustered-skewed-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.5911 | [8.4599,8.6100] | 6.242 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `wyhash w1` | 4.0784 | [3.8988,8.5328] | 6.450 |
| current-strata | `occ4-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.4284 | [8.4092,8.4408] | 6.282 |
| current-strata | `occ4-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.4231 | [8.4203,8.4238] | 6.280 |

The worst **stable** current-strata cells are `occ16-offset-beyond-direct-present-hit-*`
vs `hashtrinity_table` (10.22/10.02×, ~8.1 ns/op), reproduced in pass 2
(10.24/10.24×). The spill **absent-miss** cells that led the post-correctness
top-20 (20.5/18.2×) no longer appear; they fell to 4.39/4.44× (§8).

### 6 · Spatial failures (10 rows, none passing)

| cell | selected | incumbent | ratio | interval | concurrency | address_face | basis |
|---|---|---|---|---|---|---|---|
| `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 1.2654 | [1.1046,1.4231] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.6256 | [2.1731,2.7975] | MISMATCH | unmatched | serial |
| `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.2810 | [1.2265,1.3082] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `foldhash_par16 w16` | 2.9151 | [3.2415,3.4652] | matched | matched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `fxhash_par16 w16` | 3.0204 | [3.1397,3.4449] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.7982 | [2.5884,2.8670] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.0474 | [19.8189,20.8308] | MISMATCH | unmatched | serial |
| `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1893 | [1.1302,1.3046] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `foldhash_par16 w16` | 6.2851 | [5.8889,6.6411] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `fxhash_par16 w16` | 5.7893 | [5.4921,5.9296] | matched | matched | parallel |

The two `get_each` rows are the concurrency-unmatched (2); the four
`count_par`/`hashbrown_rayon` rows are the address-face-unmatched (4); the four
`foldhash_par16`/`fxhash_par16` rows are ratio failures. Unlike the post-correctness
run, `spatial/n100000` vs `count_par` now also exceeds the ratio limit (0.7953 →
1.2654) and fails on both gates; pass 2 measured 1.1634, also above 1.0.

### 7 · Best passing pairs (top-10 by ratio ascending)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ8-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0582 | [0.0576,0.0596] | 1.337 |
| current-strata | `occ8-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0604 | [0.0601,0.0607] | 1.334 |
| current-strata | `occ16-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0612 | [0.0611,0.0614] | 1.344 |
| current-strata | `occ1-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0614 | [0.0606,0.0622] | 1.349 |
| current-strata | `occ16-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0615 | [0.0610,0.0622] | 1.352 |
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0615 | [0.0609,0.0616] | 3.033 |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0623 | [0.0620,0.0627] | 3.011 |
| current-strata | `occ1-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0638 | [0.0617,0.0646] | 1.402 |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0642 | [0.0640,0.0643] | 3.004 |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0643 | [0.0642,0.0643] | 3.007 |

All ten are resident direct-index paths. The absent-miss rows displaced the
`existing-update` rows that led the post-correctness receipt: the resident
absent-miss selected cost fell to ~1.33–1.40 ns/op.

### 8 · Post-correctness vs post-scatter on the previously-worst cells

`pre` is the post-correctness receipt (§5/§6/§8); `post` is pass 1; `post 2` is
pass 2. Same cell, incumbent and worker width. `ns/op = sel_median_ns / 4096`.
Ratios compare the selected side to a *re-measured* incumbent, so a ratio move is
not attributable to the source alone; the incumbent ns/op is given where it moved.

#### Spill absent-miss

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash` | 20.4682 | 16.161 | 4.3922 | 3.675 | [4.3883,4.3915] | 4.5303 |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash` | 18.2012 | 15.993 | 4.4443 | 3.680 | [4.4440,4.4510] | 4.4529 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash` | 19.1104 | 18.812 | 3.3645 | 3.353 | [3.3302,3.4225] | 3.4279 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash` | 19.8637 | 18.923 | 3.4118 | 3.354 | [2.9292,3.4301] | 3.4249 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table` | 12.8636 | 18.812 | 2.4425 | 3.353 | [2.4347,2.4561] | 2.4278 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table` | 13.0705 | 18.923 | 2.4346 | 3.354 | [2.4286,2.4397] | 2.4356 |

The selected cost fell ~16.0–18.9 → ~3.35–3.68 ns/op (ratio ~18–20 → ~2.4–4.4).
This is the largest change in the receipt and reproduces across passes.

#### Spill present-hit

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 12.2664 | 9.814 | 10.2197 | 8.119 | [10.1991,10.2151] | 10.2429 |
| `occ16-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 12.3244 | 10.092 | 10.0217 | 8.133 | [9.8928,10.1629] | 10.2404 |
| `occ16-seeded-random-present-hit-replacement` | `hashtrinity_table` | 9.6735 | 10.200 | 7.9158 | 8.428 | [7.9445,8.0205] | 7.8422 |
| `occ16-seeded-random-present-hit-wrapping-add` | `hashtrinity_table` | 9.7360 | 10.280 | 8.0865 | 8.402 | [7.9468,8.1921] | 8.0126 |
| `occ8-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 14.5840 | 10.931 | 8.8709 | 6.675 | [8.8691,8.9128] | 8.8545 |
| `occ8-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 14.6136 | 10.949 | 8.8014 | 6.674 | [8.7901,8.9352] | 8.8385 |
| `occ2-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 12.1343 | 8.645 | 8.6316 | 6.172 | [8.6266,8.6557] | 8.6434 |
| `occ2-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 12.2547 | 8.727 | 8.6612 | 6.185 | [8.6271,8.6680] | 8.6329 |
| `occ4-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 12.0674 | 8.697 | 8.4284 | 6.282 | [8.4092,8.4408] | 8.4117 |
| `occ4-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 12.1534 | 8.780 | 8.4231 | 6.280 | [8.4203,8.4238] | 8.3925 |
| `occ1-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 11.8644 | 9.098 | 8.7629 | 6.450 | [8.5870,8.6119] | 8.8303 |
| `occ1-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 11.7557 | 9.009 | 8.5953 | 6.155 | [8.5658,8.8288] | 8.5996 |

The selected cost fell at every occupancy (~8.6–10.9 → ~6.2–8.4 ns/op); the
incumbent `hashtrinity_table` side is stable (~0.73–1.07 ns/op). The largest
selected-side gain is `occ8` (~10.9 → ~6.67 ns/op). The `occ1` replacement row is
order-unstable (§4a): pass 2 selected 10.339 ns/op, ratio 9.8032 vs
`hashbrown_fxhash`.

#### Growth-insert

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table` | 1.5364 | 12.561 | 2.3943 | 12.173 | [2.2984,2.5230] | 2.4066 |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table` | 1.5227 | 12.571 | 2.3358 | 12.051 | [2.3374,2.4518] | 2.4543 |

The selected cost held (~12.56 → ~12.0–12.2 ns/op). The ratio rose ~1.53 → ~2.34–2.39
because the re-measured `hashtrinity_table` incumbent fell ~8.18 → ~5.0–5.1 ns/op
(derived from `ratio` and `sel ns/op`). This is a cross-session incumbent drift, not
a selected-side regression.

#### Resident absent-miss

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-absent-miss-replacement` | `hashtrinity_table` | 4.9286 | 2.645 | 2.4699 | 1.344 | [2.4650,2.4828] | 2.4401 |
| `occ16-direct-index-resident-absent-miss-replacement` | `hashbrown_foldhash` | 1.9998 | 2.645 | 0.9616 | 1.344 | [0.9510,0.9815] | 0.9753 |

The resident absent-miss selected cost halved (~2.65 → ~1.34 ns/op); the pair against
`hashbrown_foldhash` now passes (`<= 1.0`) in both passes.

#### Spatial

| cell | selected | incumbent | pre ratio | post ratio | post interval | post 2 ratio |
|---|---|---|---|---|---|---|
| `spatial/n100000` | `w24` | `hashtrinity_count_par w24` | 0.7953 | 1.2654 | [1.1046,1.4231] | 1.1634 |
| `spatial/n100000` | `w1` | `hashtrinity_get_each w24` | 2.6974 | 2.6256 | [2.1731,2.7975] | 2.3892 |
| `spatial/n100000` | `w32` | `hashbrown_rayon w32` | 1.1167 | 1.2810 | [1.2265,1.3082] | 1.4814 |
| `spatial/n100000` | `w16` | `foldhash_par16 w16` | 2.6115 | 2.9151 | [3.2415,3.4652] | 2.7230 |
| `spatial/n100000` | `w16` | `fxhash_par16 w16` | 3.1125 | 3.0204 | [3.1397,3.4449] | 3.2467 |
| `spatial/n1000000` | `w24` | `hashtrinity_count_par w24` | 2.8653 | 2.7982 | [2.5884,2.8670] | 2.8411 |
| `spatial/n1000000` | `w1` | `hashtrinity_get_each w24` | 23.0538 | 20.0474 | [19.8189,20.8308] | 19.5996 |
| `spatial/n1000000` | `w32` | `hashbrown_rayon w32` | 1.0689 | 1.1893 | [1.1302,1.3046] | 1.1984 |
| `spatial/n1000000` | `w16` | `foldhash_par16 w16` | 5.6274 | 6.2851 | [5.8889,6.6411] | 6.2174 |
| `spatial/n1000000` | `w16` | `fxhash_par16 w16` | 6.0770 | 5.7893 | [5.4921,5.9296] | 5.9478 |

Spatial ratios are mixed and within ~±25% of the prior session except the two
`count_par` rows, which are the address-face-unmatched rows regardless of ratio.
The `n100000` selected wall cost held ~1104 → ~1096 ns across passes; the derived
incumbent is ~1104/1.2654 = ~872 ns (pass 1) and ~1096/1.1634 = ~942 ns (pass 2).
The post-correctness receipt did not record this row's selected ns, so its incumbent
side is not derivable; the ratio rise is consistent with a faster incumbent.

### 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

Pass 1:

| incumbent | pass | fail | n/a | info | post-correctness pass/fail |
|---|---|---|---|---|---|
| `hashtrinity_table` | 28 | 132 | 2 | 0 | 28 / 132 |
| `hashbrown_foldhash` | 25 | 135 | 2 | 0 | 22 / 138 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 | 0 / 160 |
| `std_siphash` | 123 | 37 | 2 | 0 | 92 / 68 |
| `xxhash_xxh3` | 152 | 8 | 2 | 0 | 144 / 16 |
| `xxhash_xxh64` | 54 | 106 | 2 | 0 | 42 / 118 |
| `wyhash` | 20 | 140 | 2 | 0 | 13 / 147 |
| `ahash` | 43 | 117 | 2 | 0 | 34 / 126 |
| `fnv` | 55 | 105 | 2 | 0 | 46 / 114 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `blake3` | 0 | 0 | 0 | 162 | 0 / 0 |
| `sha256` | 0 | 0 | 0 | 162 | 0 / 0 |

Every incumbent except `hashtrinity_table` and `hashbrown_fxhash` gained passes;
`hashtrinity_table` is unchanged at 28/132.

### 10 · Controls

55 control observations, 0 green (`observed=red`): the 19 acceptance/registration
controls (`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`), `placement-only-address` (×2),
`fabricated-current-cell` (×2), and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red). Identical set to the post-correctness receipt.

### 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above; the derived
  `ns/op = median_ns / 4096`; the pass-2 reproducibility check (1.1% status flips).
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`. The extraction identity (`ApeironWord.lean` `4c7bb34f…`) is
  recorded but its freshness was not re-run here.
- **Grade**: `measured` for the gate result; the digests and log hashes are identity
  receipts. The 55 controls establish the harness rejects corrupted observations,
  not that the table is correct. The two-pass comparison bounds, but does not
  eliminate, run-to-run variance; the host was not fully idle (§2).
- **Cell semantics**: no cell showed a semantically wrong value (wrong operation,
  wrong address face, wrong status class). The `occ1` present-hit and `xxhash_xxh3`
  swings are timing/ordering effects, not semantics; no Sol escalation triggered.

### 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair tables are retained only at the
  `/tmp` log paths with their sha256 in §2; durable retention is not included here.
  The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and
  were not captured in table mode. The `occ1` present-hit order-instability is
  characterized but not root-caused; it affects three cells and one pair in the
  HEADLINE.
- Next bounded task: attribute the stable worst current-strata ratio — the spill
  **present-hit** path `occ16-offset-beyond-direct-present-hit-*` vs
  `hashtrinity_table` (~10.2×, ~8.1 ns/op, reproduced) and `occ8-*` (~8.8×,
  ~6.7 ns/op) — to a named `src/table.rs`/`src/table/scan.rs` region, with a
  focused before/after measurement on the same 4096-op schedule. Secondary: the
  `spatial/n100000` `hashtrinity_count_par` address-face row now also exceeds the
  ratio limit. Do not weaken the `<= 1.0` contract.

## REG-1 performance remeasurement, post spill/bitmap fix (APEIRON-PERF-REMEASURE-T2)

Fresh strict-gate measurement on the current source, after the per-region Bloom
summary and the O(live) extent bitmap landed in `src/table.rs`/`src/table/scan.rs`.
The prior receipt ([`perf-ledger.md`](perf-ledger.md))
binds `src/table.rs` sha256 `a89e9ec7…`; the executing source is now
`79621926…`. The prior numbers are stale for this identity; this receipt
supersedes them for the current identity only.

### 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `79621926ba4c7b279d5511decba596fcd7b9cee06e90ea0b4525edc83fc0d5f3` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/partition.rs` | `2e9a2f56337ef1e23fe016627e3b615f456cdeb6460cbf60432118ea19d5962d` |
| `src/table/resources.rs` | `5b04be96131a360d8245ee3061c58dba85059f39a3efeeb0261442a7051368bf` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the prior receipt: `src/table.rs` (`a89e9ec7…` → `79621926…`),
`examples/reg1/acceptance.rs` (`f0f9aff0…` → `d2f8c034…`). `src/table/partition.rs`
is new to the binding table. The run writes none of these files.

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run log | `/tmp/opencode/reg1-full-postfix-20260919-172248.log`, sha256 `6ad4e40b135a3b23ba5a72e3edcf40070c6054327b27e78255d4ce2acb7bc384`, 2690 lines |

### 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`. Load average was
0.62/1.02/1.27 at start; no Lean, Kani or other build ran concurrently.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.25 (release example already cached); max RSS 553036 KiB

# 2. strict full gate (cwd /home/socol/Workspace/apeiron)
TS=$(date +%Y%m%d-%H%M%S)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  2>&1 | tee /tmp/opencode/reg1-full-postfix-$TS.log
# exit 1 (strict acceptance failure); wall 0:01:39.98; user 208.45 s; max RSS 761636 KiB
# timeout 3600 NOT reached; step 3 (smoke fallback) not run
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

### 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=387 fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 1057 are ratio failures.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

`load_bearing` rose 1458 → 2268 because `xxhash_xxh3`, `xxhash_xxh64`, `wyhash`,
`ahash` and `fnv` are now load-bearing. Raw pass/fail counts are therefore not
directly comparable to the prior receipt; the per-cell comparison in §6 is.

### 5 · Worst failing pairs (by interval max)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 30.9208 | [30.6770,31.1075] | 24.69 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 30.7018 | [30.2096,31.0015] | 24.74 |
| current-strata | `occ16-seeded-random-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 23.7463 | [23.5694,23.7792] | 19.68 |
| current-strata | `occ16-seeded-random-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 23.5900 | [23.5045,23.7635] | 19.68 |
| current-strata | `occ16-seeded-random-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 22.5222 | [20.2366,23.5354] | 24.68 |
| current-strata | `occ16-seeded-random-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 22.3041 | [21.6493,22.7078] | 24.69 |
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.7457 | [19.5026,22.2537] | n/a |
| current-strata | `occ16-seeded-random-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.6546 | [21.5898,21.8550] | 24.69 |
| current-strata | `occ16-seeded-random-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.6632 | [21.6600,21.6699] | 24.68 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.5063 | [21.4638,21.5557] | 24.74 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.4739 | [21.4177,21.5253] | 24.69 |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 19.7966 | [19.7317,19.8755] | 14.92 |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 19.7706 | [19.7261,19.8178] | 14.92 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.1096 | [17.9594,18.7543] | 24.69 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.4782 | [18.2654,18.6754] | 24.74 |
| current-strata | `occ16-seeded-random-present-hit-replacement` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.4729 | [18.2330,18.5913] | 24.68 |
| current-strata | `occ16-seeded-random-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.4102 | [17.6420,18.5549] | 24.69 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 18.1126 | [17.9611,18.1722] | 17.67 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 17.4845 | [17.4201,17.5700] | 17.61 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `wyhash w1` | 14.8015 | [14.7380,14.8657] | 24.74 |

The worst cell moved from the spill absent-miss path to the spill **present-hit**
path (`offset-beyond-direct`/`seeded-random`, ~24.7 ns/op, 30.9× worst). The
resident direct-index path is not the worst: `occ16-direct-index-resident-absent-miss-replacement`
selects 10871 ns (2.65 ns/op), ratio 4.8359 vs `hashtrinity_table` and 2.1052 vs
`hashbrown_foldhash`.

### 6 · Pre-fix vs post-fix on the previously-worst cells

Same cell, incumbent and worker width as the prior receipt §5; `pre` is
[`perf-ledger.md`](perf-ledger.md).

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash w1` | 373.8538 | 314.62 | 23.5900 | 19.68 | [23.5045,23.7635] |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 369.1557 | 310.93 | 23.7463 | 19.68 | [23.5694,23.7792] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash w1` | 365.5041 | 369.43 | 17.4845 | 17.61 | [17.4201,17.5700] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 358.6002 | 369.63 | 18.1126 | 17.67 | [17.9611,18.1722] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table w1` | 268.1054 | 369.43 | 9.5818 | 17.61 | [9.5500,9.6441] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table w1` | 265.1506 | 369.63 | 9.6489 | 17.67 | [9.5685,9.7562] |
| `occ8-direct-index-resident-admitted-fresh-insert-wrapping-add` | `hashtrinity_table w1` | 254.6132 | 550.32 | 3.4306 | 7.22 | [3.4160,3.4852] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_foldhash w1` | 251.8732 | 369.63 | 14.0958 | 17.67 | [13.9219,14.1339] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_foldhash w1` | 249.7008 | 369.43 | 13.9781 | 17.61 | [13.7856,14.1991] |
| `occ8-direct-index-resident-admitted-fresh-insert-replacement` | `hashtrinity_table w1` | 250.1037 | 554.18 | 3.5356 | 6.98 | [3.5094,3.5634] |
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table w1` | 231.8779 | 1653.21 | 1.5275 | 13.31 | [1.4823,1.6279] |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table w1` | 215.3154 | 1641.10 | 1.4905 | 12.84 | [1.4804,1.5031] |

The spill absent-miss cells fell ~15–21× in ratio and ~15–21× in absolute ns/op
(314.62 → 19.68 on the seeded-random pair); growth-insert fell ~145–152× in ratio
(1653.21 → 13.31 ns/op). The resident absent-miss pair also improved
(prior 4.6780 vs `hashtrinity_table` → now 4.8359; 1.8234 vs `hashbrown_foldhash`
→ now 2.1052). Every compared cell still fails the `<= 1.0` interval contract.

### 7 · Best passing pairs (by ratio ascending)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ16-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0840 | [0.0838,0.0843] | 3.93 |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0841 | [0.0839,0.0852] | 3.93 |
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0843 | [0.0839,0.0883] | 3.95 |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0848 | [0.0842,0.0851] | 3.96 |
| current-strata | `occ2-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0855 | [0.0854,0.0857] | 4.00 |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0855 | [0.0854,0.0861] | 4.00 |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0857 | [0.0854,0.0858] | 4.00 |
| current-strata | `occ4-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0864 | [0.0853,0.0882] | 4.05 |
| current-strata | `occ8-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0865 | [0.0862,0.0866] | 4.05 |
| current-strata | `occ8-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0866 | [0.0860,0.0867] | 4.06 |

All ten are `existing-update` on the resident direct-index path; the same cells
passed against `std_siphash` in the prior receipt. The pass set is now dominated by
`xxhash_xxh3` (152 passes), `std_siphash` (86) and `fnv` (37).

### 8 · Non-ratio failures

Concurrency-unmatched (2, both the combined `get_each` arm):

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| spatial | `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.7219 | [2.0541,3.0854] |
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.7457 | [19.5026,22.2537] |

Address-face unmatched (4):

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| spatial | `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 0.7603 | [0.6334,0.8278] |
| spatial | `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1455 | [1.0325,1.3819] |
| spatial | `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.7739 | [2.7143,2.9200] |
| spatial | `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1169 | [1.0711,1.1820] |

`n100000` vs `count_par` is under the ratio limit (0.7603) but fails the address-face
gate, so it is not a pass. The two `get_each` rows hit the concurrency check first
(`continue`) and are counted once, as concurrency-unmatched.

### 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

| incumbent | pass | fail | n/a | info |
|---|---|---|---|---|
| `hashtrinity_table` | 28 | 132 | 2 | 0 |
| `hashbrown_foldhash` | 14 | 146 | 2 | 0 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 |
| `std_siphash` | 86 | 74 | 2 | 0 |
| `xxhash_xxh3` | 152 | 8 | 2 | 0 |
| `xxhash_xxh64` | 32 | 128 | 2 | 0 |
| `wyhash` | 12 | 148 | 2 | 0 |
| `ahash` | 26 | 134 | 2 | 0 |
| `fnv` | 37 | 123 | 2 | 0 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 |
| `foldhash_par16` | 0 | 2 | 160 | 0 |
| `fxhash_par16` | 0 | 2 | 160 | 0 |
| `blake3` | 0 | 0 | 0 | 162 |
| `sha256` | 0 | 0 | 0 | 162 |

### 10 · Controls

55 control observations, 0 green (`observed=red`): the 19 acceptance/registration
controls (`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`), `placement-only-address`,
`fabricated-current-cell`, and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red; some print twice).

### 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above.
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`.
- **Grade**: `measured` for the gate result; the digests and log hash are identity
  receipts. The 55 controls establish the harness rejects corrupted observations,
  not that the table is correct.

### 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair table is retained only at the
  `/tmp` log path with the sha256 in §2; durable retention is not included here.
  The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and
  were not captured in table mode.
- Next bounded task: attribute the spill **present-hit** probe cost — the new
  worst cells `occ16-offset-beyond-direct-present-hit-{replacement,wrapping-add}`
  (30.70/30.92×, ~24.7 ns/op) and `occ16-seeded-random-present-hit-*`
  (21.7/22.5×) — to a named `src/table.rs`/`src/table/scan.rs` region, with a
  focused before/after measurement on the same 4096-op schedule. Do not weaken the
  `<= 1.0` contract.

## REG-1 performance remeasurement (APEIRON-PERF-REMEASURE-T1)

Fresh strict-gate measurement on the current source. The last matched run
([`perf-ledger.md`](perf-ledger.md)) binds `src/table.rs` sha256
`e56014da…` (pre-rework fixed-shard); the executing source is now the growable
flat/spill table at sha256 `a89e9ec7…`. Those numbers are stale; this receipt
supersedes them for the current identity only.

### 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `a89e9ec743405ac7d80183e431f6e932ba90353bc777f142865f44005261d09a` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/resources.rs` | `5b04be96131a360d8245ee3061c58dba85059f39a3efeeb0261442a7051368bf` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `f0f9aff07ff9b6feeef761f6a940386a89e3f07ba5861b39509ec4ca4e060c33` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

The run writes none of these files.

### 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run log | `/tmp/opencode/reg1-full-20260919-161738.log`, sha256 `2d4eb91b1ce8afd60d969434eaf69735faeb7a4935d952227f588efa3968ff06`, 2690 lines |

### 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`. No Lean, Kani or
other build ran concurrently; `/tmp` had 8.6 G free and load average was 1.36 at
start.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.42 (release example already cached); max RSS 542696 KiB

# 2. strict full gate (cwd /home/socol/Workspace/apeiron)
TS=$(date +%Y%m%d-%H%M%S)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  2>&1 | tee /tmp/opencode/reg1-full-$TS.log
# exit 1 (strict acceptance failure); wall 0:32:33.57; user 2335.82 s; max RSS 749488 KiB
# timeout 3600 NOT reached; step 3 (smoke fallback) not run
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

### 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=1458 applicable=650 pass=78 fail=572 missing=0 concurrency_unmatched=0 address_face_unmatched=6 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`, `concurrency_unmatched=0`;
6 failures are the separate address-face class, the other 566 are ratio failures.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.
The table-mode run prints no `acceptance_summary`/`strata_summary` (those are `--json` only).

### 5 · Worst failing pairs (by interval max)

| family | cell | selected | incumbent | ratio | interval | sel ns/op | inc ns/op |
|---|---|---|---|---|---|---|---|
| current-strata | `occ16-seeded-random-absent-miss-replacement` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 373.8538 | [370.3637,375.1818] | 314.62 | 0.84 |
| current-strata | `occ16-seeded-random-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 369.1557 | [368.1812,370.4833] | 310.93 | 0.84 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 365.5041 | [363.7280,367.1765] | 369.43 | 1.01 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 358.6002 | [356.5349,360.1007] | 369.63 | 1.03 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 268.1054 | [266.5928,269.6074] | 369.43 | 1.38 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 265.1506 | [264.0869,267.0088] | 369.63 | 1.39 |
| current-strata | `occ8-direct-index-resident-admitted-fresh-insert-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 254.6132 | [249.8611,257.7129] | 550.32 | 2.16 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashbrown_foldhash` w1 | 251.8732 | [247.8753,255.3626] | 369.63 | 1.47 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table` w1 | `hashbrown_foldhash` w1 | 249.7008 | [248.3792,253.2675] | 369.43 | 1.48 |
| current-strata | `occ8-direct-index-resident-admitted-fresh-insert-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 250.1037 | [249.2837,251.4613] | 554.18 | 2.22 |
| current-strata | `occ16-direct-index-resident-growth-insert-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 231.8779 | [209.7051,234.0361] | 1653.21 | 7.13 |
| current-strata | `occ16-direct-index-resident-growth-insert-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 215.3154 | [212.2631,233.0231] | 1641.10 | 7.62 |

The catastrophic rows are the **spill-path distributions** (`seeded-random`,
`offset-beyond-direct`) on absent-miss and the **growth-insert** cells. The resident
direct-index path is not pathological: `occ16-direct-index-resident-absent-miss-replacement`
selects 10984 ns (2.68 ns/op), ratio 4.6780 vs `hashtrinity_table` and 1.8234 vs
`hashbrown_foldhash`; `occ16-direct-index-resident-present-hit-replacement` selects
10263 ns, ratio 3.3860 vs `hashtrinity_table`.

### 6 · Best passing pairs (by ratio ascending)

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3150 | [0.3141,0.3162] |
| current-strata | `occ16-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3154 | [0.3159,0.3184] |
| current-strata | `occ8-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3168 | [0.3141,0.3185] |
| current-strata | `occ8-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3202 | [0.3163,0.3214] |
| current-strata | `occ4-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3230 | [0.3209,0.3231] |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3253 | [0.3248,0.3319] |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3269 | [0.3238,0.3278] |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3269 | [0.3265,0.3272] |
| current-strata | `occ2-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3298 | [0.3289,0.3308] |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3298 | [0.3214,0.3320] |
| current-strata | `occ2-clustered-skewed-admitted-fresh-insert-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 0.3308 | [0.3298,0.3316] |
| current-strata | `occ2-clustered-skewed-admitted-fresh-insert-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 0.3323 | [0.3318,0.3343] |

All 78 passes are `existing-update` or `clustered-skewed` cells against `std_siphash`
or `hashtrinity_table`.

### 7 · Address-face unmatched (6 pairs, fail regardless of ratio)

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| spatial | `spatial/n100000` | `apeiron_table` w24 | `hashtrinity_count_par` w24 | 1.6967 | [1.6297,1.7154] |
| spatial | `spatial/n100000` | `apeiron_table` w1 | `hashtrinity_get_each` w1 | 7.4258 | [7.1715,8.0202] |
| spatial | `spatial/n100000` | `apeiron_table` w32 | `hashbrown_rayon` w32 | 1.6991 | [1.6196,1.7571] |
| spatial | `spatial/n1000000` | `apeiron_table` w24 | `hashtrinity_count_par` w24 | 6.8355 | [6.8527,6.9322] |
| spatial | `spatial/n1000000` | `apeiron_table` w1 | `hashtrinity_get_each` w1 | 80.2115 | [79.2462,81.7946] |
| spatial | `spatial/n1000000` | `apeiron_table` w32 | `hashbrown_rayon` w32 | 2.5814 | [2.4410,2.7745] |

The two `par16` spatial incumbents are address-face `matched` and fail on ratio:
`foldhash_par16` 4.3537 / 18.8233 and `fxhash_par16` 5.4138 / 17.2777 at
`n100000` / `n1000000`.

### 8 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

| incumbent | pass | fail | n/a | baseline pass/fail |
|---|---|---|---|---|
| `hashtrinity_table` | 20 | 140 | 2 | 14 / 146 |
| `hashbrown_foldhash` | 0 | 160 | 2 | 11 / 149 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 10 / 150 |
| `std_siphash` | 58 | 102 | 2 | 98 / 62 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 / 2 |

Informational incumbents (`xxhash_xxh3`, `xxhash_xxh64`, `wyhash`, `ahash`, `fnv`,
`blake3`, `sha256`): 162 info rows each, excluded from the load-bearing pass.

Delta vs [`perf-ledger.md`](perf-ledger.md) (different `src/table.rs`
digest, and its harness lacked the address-face gate): aggregate worst serial ratio
rose from `r_max=29.5121` to `375.1818`, both against `hashbrown_fxhash`, on different
absent-miss cells. Total applicable passes fell 133 → 78. The regression tracks the
growable spill/growth path, not the resident path.

### 9 · Controls

19/19 controls red (`child_exit=2`), 0 green: `wrong-workload-identity`,
`invalid-samples`, `absent-samples`, `cached-home-only-collision`,
`wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`. Additional registration controls
(`placement-only-address`, `retired-shardfull-{receipt,digest}` ×16) also red.

### 10 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine, toolchain
  and dirty-source identity; the 2592-row pair table; the 19 controls; the fail count
  and the worst/best pairs above.
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8` identity
  row, cross-process determinism, codec round-trip, `remove`, SIMD, allocator
  behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the unextracted
  `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`.
- **Grade**: `measured` for the gate result; the digests and log hash are identity
  receipts. The 19 controls establish the harness rejects corrupted observations,
  not that the table is correct.

### 11 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair table is retained only at the `/tmp`
  log path with the sha256 in §2; durable retention is not included here. The
  per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and were not
  captured in table mode.
- Next bounded task: attribute the spill-path and growth-insert regression to a named
  `src/table.rs` code region (the `seeded-random`/`offset-beyond-direct` absent-miss and
  `growth-insert` cells at 250–375× and 215–232×), with a focused before/after
  measurement on the same 4096-op schedule. Do not weaken the `<= 1.0` contract.

## First-cut baseline receipt (APEIRON-PERF-REWORK-W3-T1)

Binding: this receipt binds to the dirty-tree digests below, **not to HEAD**.
`apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`. The captured
`src/table.rs` is uncommitted; HEAD's copy differs. Any later W4-T1 rewrite is
compared against the digests in §1, not against a commit.

### 1 · First-cut source snapshot

| artifact | sha256 | lines |
|---|---|---|
| live `src/table.rs` (at capture) | `c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` | 389 |
| snapshot `docs/evidence/first-cut/table.rs` | `c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` | 389 |
| live `tests/table.rs` (at capture) | `cf9969bf0c2ededd97cb23c6a36f7abfcd9614f3c2b7e84595f97962ecf10641` | 59 |
| snapshot `docs/evidence/first-cut/tests-table.rs` | `cf9969bf0c2ededd97cb23c6a36f7abfcd9614f3c2b7e84595f97962ecf10641` | 59 |

`cmp` reports both snapshot pairs byte-identical to the live files at capture
(`TABLE_SNAPSHOT_IDENTICAL`, `TESTS_SNAPSHOT_IDENTICAL`). Re-checked after the
strict run: live digests unchanged.

### 2 · Harness identity at capture

| artifact | sha256 | lines |
|---|---|---|
| `examples/reg1/main.rs` | `0e132290a0b084a03fb99feac977de6d0f6fe910b62fae7693a65adb8f735dd7` | 606 |
| `examples/reg1/strata.rs` | `895ff8ef65b2a6e53b81bdd31647efd8659a6783187eb8a09fa9c3a399fb0d3b` | 1490 |
| `examples/reg1/performance.rs` | `bd73bb70116dc798c007838d7764164c07c5270f0530ed74248ccc076d472238` | 1182 |
| `examples/reg1/matrix.rs` | `620eae3ef85390f18f840937ce96a0cc606d9fb2c06e63a112eaa3459c6095b1` | 1385 |
| `examples/reg1/workload.rs` | `100371053e8e00a50eb91e90887c41659996aa30b6ee6392a8a4493e1d63cf5d` | 817 |
| `examples/reg1/correctness.rs` | `4294f5ec2a6172d76c8e893ccf31973aaf5a7b06d5c839b02d3edd4473e62379` | 553 |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` | 1636 |

`matrix.json` re-checked after the strict run: digest unchanged; the run does not
write it.

### 3 · Prior before-measurements (handoff-supplied)

Recorded as stated by the W3-T1 handoff; no in-repo citation for these numbers was
located, so they are prior measurements with source not in this tree.

| configuration | ratio (replacement / wrapping-add) | n |
|---|---|---|
| historical `CountTable` / hashtrinity | `0.2813` | 1M |
| bounded fixed-16 shared witness | `2.4647` / `2.4286` | — |
| first growable cut | `9.7270` / `9.6198` | — |

The fresh strict run (§4) emits first-cut shared fixed-16 ratios `9.5316` /
`9.5121`. The two first-cut pairs differ across measurement moments; the fresh
run's numbers are authoritative for this receipt's source digest.

### 4 · Strict acceptance run on the first-cut source

Command (cwd `/home/socol/Workspace/apeiron`):

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Exit code: **1** (strict acceptance failure; the run is the evidence, nothing was fixed).
A cargo warning `error: duplicate key` from the hashtrinity git checkout
(`opsfold/Cargo.toml:61`) is printed before `Finished`; it is non-fatal and does
not determine the exit status.

`strata_summary`:

```text
strata_summary cells=160 measured=160 pass=24 fail=136 not_applicable=0 missing=0 rejected=0 resource_cells=3 resource_not_applicable=1 resource_missing=2 campaigns=3 warmup=2 samples=15 ops=4096 ht_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc="rustc 1.97.0-nightly (17584a181 2026-04-13)" cpu="Intel(R) Core(TM) i9-14900KF"
```

`strict_acceptance` (from the `axis=performance` evidence line):

```text
strict_acceptance=fail applicable=160 failing=136 worst_cell=occ16-direct-index-resident-absent-miss-wrapping-add ratio=45.4000
```

| field | value |
|---|---|
| failing cells | 136 / 160 |
| worst cell | `occ16-direct-index-resident-absent-miss-wrapping-add` |
| worst ratio | `45.4000` |
| ratio limit | `1.00` |
| first failing cell | `occ1-direct-index-resident-present-hit-replacement` ratio `4.2256` |
| shared fixed-16 axis | replacement `9.5316`, wrapping-add `9.5121` |
| candidate/hashtrinity | `2.5810` (experimental, not the REG-1 axis) |
| candidate/current | `0.2713` (experimental, not the REG-1 axis) |

Axis verdicts: `identity=pass`, `round_trip=boundary`, `determinism=missing`,
`growth_merge_refusal=pass`, `performance=fail`, `coverage=missing`.
`report_complete=false`, `reg1_pass=false`.

### 5 · Controls

Observed: **25 control lines, 25 red, 0 green** (11 distinct control names:
`retired-shardfull-receipt` ×8, `retired-shardfull-digest` ×8,
`fabricated-current-cell`, `wrong-workload-identity`, `invalid-samples`,
`absent-samples`, `cached-home-only-collision`, `wrong-oracle-digest`,
`deleted-required-current-cell`, `selected-implementation-mismatch`,
`strict-acceptance-single-failing-cell`).

The W3-T1 handoff expected `8/8` red controls. That count is **not** reproduced by
this source revision; the observed count is 25/25. Recorded as observed; the
handoff figure is not adopted.

### 6 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux (rolling), kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| apeiron head | `c450c222905c050f793a23332708c230c658f82c` (dirty) |

Raw run log: `/tmp/opencode/reg1-strict-baseline.log`,
sha256 `28268f42821ec47444d121918a722fdd8fc36a697e912cc85b31a4260d6a9afe`
(ephemeral; the exact lines above are the durable record).

### 7 · Scope and exclusions

This receipt is the "before" for the W4-T1 rewrite. It captures source identity
and a red strict-acceptance result only.

- In scope: identity of the first-cut `Table`, the harness, and the strict gate
  result on this machine/toolchain.
- Out of scope (section-8 obligations not evidenced here): `Sigma` in every
  executing signature, `remove`, SIMD, allocator behaviour, threading, std
  internals, rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V.
- No history, Kani, `proof/regen.sh`, or `lake build` was run for this receipt.
- Grade: snapshot and harness identity are digest receipts (`tested` identity);
  the strict run is a `measured`/gate result. The prior numbers in §3 are
  handoff-supplied prior measurements, source not located.

### 8 · Open items

- **REG-1 OPEN**: the strict performance acceptance fails 136/160 on this source.
- **G1 OPEN**: the performance objective is not met by the first growable cut.
