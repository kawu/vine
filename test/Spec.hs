import           Test.Tasty (TestTree, testGroup)
import qualified Test.Tasty as Tasty
import qualified Test.Tasty.SmallCheck as SC

import qualified Data.Map.Strict as M

import qualified Net.Graph2 as N
import qualified Net.Graph2.BiComp as B


main :: IO ()
main = Tasty.defaultMain tests


tests :: TestTree
tests = testGroup "Tests" [properties] --, unitTests]


properties :: TestTree
properties = testGroup "Properties" [scProps] -- , qcProps]


scProps = testGroup "(checked by SmallCheck)"
--   [ SC.testProperty "sort == sort . reverse" $
--       \list -> sort (list :: [Int]) == sort (reverse list)
  [ SC.testProperty "rightInTwo (xs ++ xs) == (xs, xs)" $
      \xs -> N.rightInTwo (xs ++ xs :: [Int]) == (xs, xs)
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "(decode . encode) x == x" $
      \x -> (N.decode . N.encode) x == x
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "(explicate . obfuscate) x == x" $
      \xs ->
        let m = M.fromList (zip N.enumerate $ pad 8 xs)
         in null xs || (N.explicate . N.obfuscate) m == m
  , SC.testProperty "explicate (mask x) M.! y == [x == y]" $
      \x y -> N.explicate (B.Vec $ N.mask x) M.! y ==
        if x == y then 1.0 else 0.0
  ]


-- | Pad the given (non-empty) list to the given number of elements.
pad :: Int -> [a] -> [a]
pad k = take k . cycle
