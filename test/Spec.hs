import           Test.Tasty (TestTree, testGroup)
import qualified Test.Tasty as Tasty
import qualified Test.Tasty.SmallCheck as SC

-- import           Data.List (sort)

import qualified Net.Graph2 as N


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
      \x -> N.decode (N.encode x) == x
  ]
