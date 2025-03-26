ffmpeg -framerate 23 -i opt_%d_csg_wo_volume.png -c:v libx264 -r 30 -pix_fmt yuv420p output_csg_novolume.mp4

ffmpeg -i output_csg_novolume.mp4 -i output_csg_volume.mp4 -filter_complex "[0:v]crop=768:288:0:0[v0];[1:v]scale=768:-1,crop=768:288:0:288[v1]; [v0][v1]vstack[v]" -map [v] -vcodec libx264 -pix_fmt yuv420p -preset ultrafast out_put_csg_bc.mp4