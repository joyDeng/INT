ffmpeg -framerate 30 -i Btwo_layer_opt_%03d_mid_plane.png -c:v libx264 -r 30 -pix_fmt yuv420p two_layer_vc_5s.mp4

ffmpeg -i output_csg_constraint_energy.mp4 -i output_csg_volume_non_sym_emitter.mp4 -filter_complex "[0:v]crop=1024:512:0:0[v0];[1:v]scale=1024:-1,crop=1024:512:0:512[v1]; [v0][v1]vstack[v]" -map [v] -vcodec libx264 -pix_fmt yuv420p -preset ultrafast stellarator.mp4