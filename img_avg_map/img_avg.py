def avg_frame(pma_file_path):
    try:
        Frames_data = read_pma(pma_file_path)
        avg_frame_data = np.mean(Frames_data, axis=0)
        return avg_frame_data

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None