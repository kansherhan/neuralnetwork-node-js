class Utils
{
    static argmax(arr: Array<any>): number {
        let maxValueIndex = 0;
        let maxValue = 0;

        for (let i = 0; i < arr.length; i++) {
			if (arr[i] > maxValue)
			{
				maxValue = arr[i];
				maxValueIndex = i;
			}
		}

        return maxValueIndex;
    }
}

export { Utils };